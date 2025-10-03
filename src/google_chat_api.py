"""
Google API Client - Handles all communication with Google's Gemini API.
This module is used by both OpenAI compatibility layer and native Gemini endpoints.
"""
import asyncio
import gc
import json
from typing import Optional

from fastapi import Response
from fastapi.responses import StreamingResponse

from config import (
    get_code_assist_endpoint,
    DEFAULT_SAFETY_SETTINGS,
    get_base_model_name,
    get_thinking_budget,
    should_include_thoughts,
    is_search_model,
    get_auto_ban_enabled,
    get_auto_ban_error_codes,
    get_auto_ban_keywords,
    get_retry_429_max_retries,
    get_retry_429_enabled,
    get_retry_429_interval
)
from .httpx_client import http_client, create_streaming_client_with_kwargs
from log import log
from .credential_manager import CredentialManager
from .usage_stats import record_successful_call
from .utils import get_user_agent

def _create_error_response(message: str, status_code: int = 500) -> Response:
    """Create standardized error response."""
    return Response(
        content=json.dumps({
            "error": {
                "message": message,
                "type": "api_error",
                "code": status_code
            }
        }),
        status_code=status_code,
        media_type="application/json"
    )

def _extract_error_message(status_code: int, response_content: str = "") -> str:
    """Return a concise error message from raw response content."""
    fallback_messages = {
        429: "429 rate limit exceeded"
    }
    if response_content:
        content = response_content.strip()
        if content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    error_block = parsed.get('error')
                    if isinstance(error_block, dict):
                        message = error_block.get('message')
                        if isinstance(message, str) and message.strip():
                            return message.strip()
                    if 'message' in parsed and isinstance(parsed['message'], str):
                        message = parsed['message'].strip()
                        if message:
                            return message
            except Exception:
                pass
            if len(content) > 500:
                content = content[:500]
            return content
    return fallback_messages.get(status_code, f"API error: {status_code}")


async def _should_auto_ban(status_code: int, response_content: str) -> tuple[bool, Optional[str]]:
    """Evaluate auto-ban rules and return (triggered, reason)."""
    if not await get_auto_ban_enabled():
        return False, None

     # 并发获取，降低一次网络/IO 往返
    codes, keywords_map = await asyncio.gather(
        get_auto_ban_error_codes(),
        get_auto_ban_keywords(),
    )

    # 若配置了 codes 且当前 status_code 不在其中，提前返回
    if codes and status_code not in codes:
        return False, None


    # 收集与本 status_code 生效的关键字（或通配符 *），或列表形式
    keywords_to_check: list[str] = []
    if isinstance(keywords_map, dict) and keywords_map:
        kw = keywords_map.get(str(status_code)) or keywords_map.get("*") or []
        keywords_to_check.extend([k for k in kw if k])
    elif isinstance(keywords_map, list) and keywords_map:
        keywords_to_check.extend([k for k in keywords_map if k])

    # 如果配置了关键字，则“必须命中关键字”才触发
    if keywords_to_check:
        if not response_content:
            return False, None
        normalized = response_content.lower()
        # 预处理关键字，strip + lower
        for raw in keywords_to_check:
            kw = raw.strip().lower()
            if kw and kw in normalized:
                return True, f"keyword '{kw}'"
        return False, None

    # 未配置关键字 => 仅凭状态码触发（若 codes 为空，表示不启用状态码触发）
    if codes:
        return True, f"status {status_code}"

    return False, None


async def _handle_api_error(credential_manager: CredentialManager, status_code: int, response_content: str = "", current_file: str = None):
    """
    Handle API errors by checking auto-ban rules and disabling credentials if needed.
    Note: 429 retry logic is handled separately in the request functions.
    """
    if not credential_manager:
        return
    
    auto_ban_triggered, trigger_reason = await _should_auto_ban(status_code, response_content)

    if auto_ban_triggered:
        if trigger_reason:
            if response_content:
                log.error("Google API returned %s - auto ban triggered (%s). Response details: %s" % (status_code, trigger_reason, response_content[:500]))
            else:
                log.warning("Google API returned %s - auto ban triggered (%s)" % (status_code, trigger_reason))
        
        # 尝试禁用当前凭证，如果失败则强制轮换
        if current_file:
            disabled_success = await credential_manager.set_cred_disabled(current_file, True)
            if not disabled_success:
                await credential_manager.force_rotate_credential()
        else:
            await credential_manager.force_rotate_credential()





async def _prepare_request_headers_and_payload(payload: dict, credential_data: dict):
    """Prepare request headers and final payload from credential data."""
    # 尝试获取token，支持多种字段名
    token = credential_data.get('token') or credential_data.get('access_token', '')
    
    if not token:
        raise Exception("凭证中没有找到有效的访问令牌（token或access_token字段）")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    # 直接使用凭证数据中的项目ID
    project_id = credential_data.get("project_id", "")
    if not project_id:
        raise Exception("项目ID不存在于凭证数据中")

    final_payload = {
        "model": payload.get("model"),
        "project": project_id,
        "request": payload.get("request", {})
    }
    
    return headers, final_payload


async def _handle_response_error(
    status_code: int,
    response_content: str,
    credential_manager: CredentialManager,
    current_file: str,
    is_streaming: bool = False
) -> None:
    """统一处理API响应错误（包括429和其他错误）"""
    # 记录详细错误信息
    stream_type = "STREAMING" if is_streaming else "NON-STREAMING"
    if response_content:
        log.error(f"Google API returned status {status_code} ({stream_type}). Response details: {response_content[:500]}")
    else:
        log.error(f"Google API returned status {status_code} ({stream_type})")
    
    # 记录API调用失败
    if credential_manager and current_file:
        error_message = _extract_error_message(status_code, response_content)
        await credential_manager.record_api_call_result(current_file, False, status_code, error_message)
    
    # 统一调用错误处理（包括429的自动禁用）
    await _handle_api_error(credential_manager, status_code, response_content, current_file)

async def send_gemini_request(
    payload: dict, 
    is_streaming: bool = False, 
    credential_manager: CredentialManager = None,
    credential_file: str = None,
    credential_data: dict = None
) -> Response:
    """
    Send a request to Google's Gemini API.
    
    Args:
        payload: The request payload in Gemini format
        is_streaming: Whether this is a streaming request
        credential_manager: CredentialManager instance
        credential_file: The credential filename (should be provided by caller)
        credential_data: The credential data dict (should be provided by caller)
        
    Returns:
        FastAPI Response object
    """
    # 获取429重试配置
    max_retries = await get_retry_429_max_retries()
    retry_429_enabled = await get_retry_429_enabled()
    retry_interval = await get_retry_429_interval()
    
    # 确定API端点
    action = "streamGenerateContent" if is_streaming else "generateContent"
    target_url = f"{await get_code_assist_endpoint()}/v1internal:{action}"
    if is_streaming:
        target_url += "?alt=sse"

    # 确保有credential_manager和凭证数据
    if not credential_manager:
        return _create_error_response("Credential manager not provided", 500)
    
    # 优先使用传入的凭证，如果没有则获取新的
    if not credential_file or not credential_data:
        try:
            credential_result = await credential_manager.get_valid_credential()
            if not credential_result:
                return _create_error_response("No valid credentials available", 500)
            
            credential_file, credential_data = credential_result
        except Exception as e:
            return _create_error_response(str(e), 500)
    
    current_file = credential_file
    
    # 准备请求头和payload
    try:
        headers, final_payload = await _prepare_request_headers_and_payload(payload, credential_data)
    except Exception as e:
        return _create_error_response(str(e), 500)

    # 预序列化payload，避免重试时重复序列化
    final_post_data = json.dumps(final_payload)
    
    # Debug日志：打印请求体结构
    log.debug(f"Final request payload structure: {json.dumps(final_payload, ensure_ascii=False, indent=2)}")

    # 重试循环
    for attempt in range(max_retries + 1):
        try:
            if is_streaming:
                return await _send_streaming_request(
                    target_url, final_post_data, headers, 
                    credential_manager, current_file, payload.get("model", ""),
                    retry_429_enabled, attempt, max_retries, retry_interval,
                    payload, credential_data
                )
            else:
                return await _send_non_streaming_request(
                    target_url, final_post_data, headers,
                    credential_manager, current_file, payload.get("model", ""),
                    retry_429_enabled, attempt, max_retries, retry_interval,
                    payload, credential_data
                )
                    
        except Exception as e:
            if attempt < max_retries:
                log.warning(f"[RETRY] Request failed with exception, retrying ({attempt + 1}/{max_retries}): {str(e)}")
                await asyncio.sleep(retry_interval)
                continue
            else:
                log.error(f"Request to Google API failed: {str(e)}")
                return _create_error_response(f"Request failed: {str(e)}")
    
    # 如果循环结束仍未成功，返回错误
    return _create_error_response("Max retries exceeded", 429)


async def _send_streaming_request(
    target_url: str, 
    final_post_data: str, 
    headers: dict,
    credential_manager: CredentialManager,
    current_file: str,
    model_name: str,
    retry_429_enabled: bool,
    attempt: int,
    max_retries: int,
    retry_interval: float,
    original_payload: dict,
    credential_data: dict
) -> Response:
    """发送流式请求的内部函数"""
    client = await create_streaming_client_with_kwargs()
    
    try:
        stream_ctx = client.stream("POST", target_url, content=final_post_data, headers=headers)
        resp = await stream_ctx.__aenter__()
        
        # 检查响应状态码
        if resp.status_code != 200:
            # 读取错误响应内容
            response_content = ""
            try:
                content_bytes = await resp.aread()
                if isinstance(content_bytes, bytes):
                    response_content = content_bytes.decode('utf-8', errors='ignore')
            except Exception as e:
                log.debug(f"Failed to read error response content: {e}")
            
            # 清理资源
            try:
                await stream_ctx.__aexit__(None, None, None)
            except:
                pass
            await client.aclose()
            
            # 统一处理错误（包括429）
            await _handle_response_error(resp.status_code, response_content, credential_manager, current_file, is_streaming=True)
            
            # 如果是429且允许重试
            if resp.status_code == 429 and retry_429_enabled and attempt < max_retries:
                log.warning(f"[RETRY] 429 error encountered, retrying ({attempt + 1}/{max_retries})")
                if credential_manager:
                    # 重新获取凭证
                    new_credential_result = await credential_manager.get_valid_credential()
                    if new_credential_result:
                        new_file, new_credential_data = new_credential_result
                        new_headers, updated_payload = await _prepare_request_headers_and_payload(original_payload, new_credential_data)
                        new_post_data = json.dumps(updated_payload)
                        await asyncio.sleep(retry_interval)
                        # 递归重试（通过抛出异常让外层循环处理）
                        raise Exception("429 retry needed")
            
            # 返回错误流
            async def error_stream():
                error_response = {
                    "error": {
                        "message": _extract_error_message(resp.status_code, response_content),
                        "type": "api_error",
                        "code": resp.status_code
                    }
                }
                yield f"data: {json.dumps(error_response)}\n\n"
            return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=resp.status_code)
        
        # 成功响应，传递给流式处理函数
        return _handle_streaming_response_managed(resp, stream_ctx, client, credential_manager, model_name, current_file)
        
    except Exception as e:
        # 清理资源
        try:
            await client.aclose()
        except:
            pass
        raise e


async def _send_non_streaming_request(
    target_url: str,
    final_post_data: str,
    headers: dict,
    credential_manager: CredentialManager,
    current_file: str,
    model_name: str,
    retry_429_enabled: bool,
    attempt: int,
    max_retries: int,
    retry_interval: float,
    original_payload: dict,
    credential_data: dict
) -> Response:
    """发送非流式请求的内部函数"""
    async with http_client.get_client(timeout=None) as client:
        resp = await client.post(target_url, content=final_post_data, headers=headers)
        
        # 检查响应状态码
        if resp.status_code != 200:
            # 读取错误响应内容
            response_content = ''
            try:
                response_content = resp.text or ''
            except Exception:
                response_content = ''
            
            # 统一处理错误（包括429）
            await _handle_response_error(resp.status_code, response_content, credential_manager, current_file, is_streaming=False)
            
            # 如果是429且允许重试
            if resp.status_code == 429 and retry_429_enabled and attempt < max_retries:
                log.warning(f"[RETRY] 429 error encountered, retrying ({attempt + 1}/{max_retries})")
                if credential_manager:
                    # 重新获取凭证
                    new_credential_result = await credential_manager.get_valid_credential()
                    if new_credential_result:
                        new_file, new_credential_data = new_credential_result
                        new_headers, updated_payload = await _prepare_request_headers_and_payload(original_payload, new_credential_data)
                        new_post_data = json.dumps(updated_payload)
                        await asyncio.sleep(retry_interval)
                        # 递归重试（通过抛出异常让外层循环处理）
                        raise Exception("429 retry needed")
            
            return _create_error_response(_extract_error_message(resp.status_code, response_content), resp.status_code)
        
        # 成功响应
        return await _handle_non_streaming_response(resp, credential_manager, model_name, current_file)


def _handle_streaming_response_managed(resp, stream_ctx, client, credential_manager: CredentialManager = None, model_name: str = "", current_file: str = None) -> StreamingResponse:
    """Handle streaming response with complete resource lifecycle management."""
    
    # 正常流式响应处理，确保资源在流结束时被清理
    async def managed_stream_generator():
        success_recorded = False
        chunk_count = 0
        try:
            async for chunk in resp.aiter_lines():
                if not chunk or not chunk.startswith('data: '):
                    continue
                    
                # 记录第一次成功响应
                if not success_recorded:
                    if current_file and credential_manager:
                        await credential_manager.record_api_call_result(current_file, True)
                        # 记录到使用统计
                        try:
                            await record_successful_call(current_file, model_name)
                        except Exception as e:
                            log.debug(f"Failed to record usage statistics: {e}")
                    success_recorded = True
                
                # 原样返回官方响应，chunk已经是字符串格式
                # aiter_lines()返回的是str，需要编码为bytes + SSE格式的\n\n
                yield f"{chunk}\n\n".encode()
                await asyncio.sleep(0)  # 让其他协程有机会运行
                
                # 定期释放内存（每100个chunk）
                chunk_count += 1
                if chunk_count % 100 == 0:
                    gc.collect()
                    
        except Exception as e:
            log.error(f"Streaming error: {e}")
            err = {"error": {"message": str(e), "type": "api_error", "code": 500}}
            yield f"data: {json.dumps(err)}\n\n".encode()
        finally:
            # 确保清理所有资源
            try:
                await stream_ctx.__aexit__(None, None, None)
            except Exception as e:
                log.debug(f"Error closing stream context: {e}")
            try:
                await client.aclose()
            except Exception as e:
                log.debug(f"Error closing client: {e}")

    return StreamingResponse(
        managed_stream_generator(),
        media_type="text/event-stream"
    )

async def _handle_non_streaming_response(resp, credential_manager: CredentialManager = None, model_name: str = "", current_file: str = None) -> Response:
    """Handle non-streaming response from Google API."""
    if resp.status_code == 200:
        # 记录成功响应
        if current_file and credential_manager:
            await credential_manager.record_api_call_result(current_file, True)
            # 记录到使用统计
            try:
                await record_successful_call(current_file, model_name)
            except Exception as e:
                log.debug(f"Failed to record usage statistics: {e}")
        
        # 原样返回官方响应,不做修改或检查
        raw = await resp.aread()
        if log.level <= 10:  # DEBUG level
            try:
                google_api_response = raw.decode('utf-8')
                if google_api_response.startswith('data: '):
                    google_api_response = google_api_response[len('data: '):]
                google_api_response = json.loads(google_api_response)
                log.debug(f"Google API原始响应: {json.dumps(google_api_response, ensure_ascii=False)[:500]}...")
            except Exception as e:
                log.debug(f"Failed to decode response for logging: {e}")
        
        return Response(
            content=raw,
            status_code=200,
            media_type=resp.headers.get("Content-Type", "application/json; charset=utf-8")
        )
    else:
        # 错误响应不应该到这里,因为在_send_non_streaming_request中已经处理了
        # 但为了安全起见,仍然保留这个分支
        log.error(f"Unexpected error response in _handle_non_streaming_response: {resp.status_code}")
        return _create_error_response(f"API error: {resp.status_code}", resp.status_code)

def build_gemini_payload_from_native(native_request: dict, model_from_path: str) -> dict:
    """
    Build a Gemini API payload from a native Gemini request with full pass-through support.
    """
    # 创建请求副本以避免修改原始数据
    request_data = native_request.copy()
    
    # 应用默认安全设置（如果未指定）
    if "safetySettings" not in request_data:
        request_data["safetySettings"] = DEFAULT_SAFETY_SETTINGS


    # 限制参数
    if "generationConfig" in request_data and request_data["generationConfig"]:
        generation_config = request_data["generationConfig"]
        
        # 限制max_tokens (在Gemini中叫maxOutputTokens)
        if "maxOutputTokens" in generation_config and generation_config["maxOutputTokens"] is not None:
            if generation_config["maxOutputTokens"] > 65535:
                generation_config["maxOutputTokens"] = 65535
                
        # 覆写 top_k 为 64 (在Gemini中叫topK)
        # generation_config["topK"] = 64
    else:
        # 如果没有generationConfig，创建一个并设置topK
        # request_data["generationConfig"] = {"topK": 64}
        pass
    
    # 确保generationConfig存在
    if "generationConfig" not in request_data:
        request_data["generationConfig"] = {}
    
    generation_config = request_data["generationConfig"]
    
    # 配置thinking（如果未指定thinkingConfig）
    if "thinkingConfig" not in generation_config:
        generation_config["thinkingConfig"] = {}
    
    thinking_config = generation_config["thinkingConfig"]
    
    # 只有在未明确设置时才应用默认thinking配置
    if "includeThoughts" not in thinking_config:
        thinking_config["includeThoughts"] = should_include_thoughts(model_from_path)
    if "thinkingBudget" not in thinking_config:
        thinking_config["thinkingBudget"] = get_thinking_budget(model_from_path)
    
    # 为搜索模型添加Google Search工具（如果未指定且没有functionDeclarations）
    if is_search_model(model_from_path):
        if "tools" not in request_data:
            request_data["tools"] = []
        # 检查是否已有functionDeclarations或googleSearch工具
        has_function_declarations = any(tool.get("functionDeclarations") for tool in request_data["tools"])
        has_google_search = any(tool.get("googleSearch") for tool in request_data["tools"])
        
        # 只有在没有任何工具时才添加googleSearch，或者只有googleSearch工具时可以添加更多googleSearch
        if not has_function_declarations and not has_google_search:
            request_data["tools"].append({"googleSearch": {}})
    
    # 透传所有其他Gemini原生字段:
    # - contents (必需)
    # - systemInstruction (可选)
    # - generationConfig (已处理)
    # - safetySettings (已处理)  
    # - tools (已处理)
    # - toolConfig (透传)
    # - cachedContent (透传)
    # - 以及任何其他未知字段都会被透传
    
    return {
        "model": get_base_model_name(model_from_path),
        "request": request_data
    }