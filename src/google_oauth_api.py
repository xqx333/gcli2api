"""
Google OAuth2 认证模块
"""
import time
import jwt
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from urllib.parse import urlencode

from config import get_oauth_proxy_url, get_googleapis_proxy_url, get_resource_manager_api_url, get_service_usage_api_url
from log import log
from .httpx_client import get_async, post_async


class TokenError(Exception):
    """Token相关错误"""
    pass

class Credentials:
    """凭证类"""
    
    def __init__(self, access_token: str, refresh_token: str = None,
                 client_id: str = None, client_secret: str = None,
                 expires_at: datetime = None, project_id: str = None):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.client_id = client_id
        self.client_secret = client_secret
        self.expires_at = expires_at
        self.project_id = project_id
        
        # 反代配置将在使用时异步获取
        self.oauth_base_url = None
        self.token_endpoint = None
    
    def is_expired(self) -> bool:
        """检查token是否过期"""
        if not self.expires_at:
            return True
        
        # 提前3分钟认为过期
        buffer = timedelta(minutes=3)
        return (self.expires_at - buffer) <= datetime.now(timezone.utc)
    
    async def refresh_if_needed(self) -> bool:
        """如果需要则刷新token"""
        if not self.is_expired():
            return False
        
        if not self.refresh_token:
            raise TokenError("需要刷新令牌但未提供")
        
        await self.refresh()
        return True
    
    async def refresh(self, max_retries: int = 3, base_delay: float = 1.0):
        """刷新访问令牌，支持重试机制"""
        if not self.refresh_token:
            raise TokenError("无刷新令牌")
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token'
        }
        
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                oauth_base_url = await get_oauth_proxy_url()
                token_url = f"{oauth_base_url.rstrip('/')}/token"
                response = await post_async(
                    token_url,
                    data=data,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                )
                response.raise_for_status()
                
                token_data = response.json()
                self.access_token = token_data['access_token']
                
                if 'expires_in' in token_data:
                    expires_in = int(token_data['expires_in'])
                    self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
                
                if 'refresh_token' in token_data:
                    self.refresh_token = token_data['refresh_token']
                
                if attempt > 0:
                    log.debug(f"Token刷新成功（第{attempt + 1}次尝试），过期时间: {self.expires_at}")
                else:
                    log.debug(f"Token刷新成功，过期时间: {self.expires_at}")
                return
                
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                
                # 检查是否是不可恢复的错误，如果是则不重试
                if self._is_non_retryable_error(error_msg):
                    log.error(f"Token刷新遇到不可恢复错误: {error_msg}")
                    break
                
                if attempt < max_retries:
                    # 计算退避延迟时间（指数退避）
                    delay = base_delay * (2 ** attempt)
                    log.warning(f"Token刷新失败（第{attempt + 1}次尝试）: {error_msg}，{delay}秒后重试...")
                    await asyncio.sleep(delay)
                else:
                    break
        
        # 所有重试都失败了
        error_msg = f"Token刷新失败（已重试{max_retries}次）: {str(last_exception)}"
        log.error(error_msg)
        raise TokenError(error_msg)
    
    def _is_non_retryable_error(self, error_msg: str) -> bool:
        """判断是否是不需要重试的错误"""
        non_retryable_patterns = [
            "400 Bad Request",
            "invalid_grant",
            "refresh_token_expired",
            "invalid_refresh_token", 
            "unauthorized_client",
            "access_denied",
            "401 Unauthorized"
        ]
        
        error_msg_lower = error_msg.lower()
        for pattern in non_retryable_patterns:
            if pattern.lower() in error_msg_lower:
                return True
                
        return False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Credentials':
        """从字典创建凭证"""
        # 处理过期时间
        expires_at = None
        if 'expiry' in data and data['expiry']:
            try:
                expiry_str = data['expiry']
                if isinstance(expiry_str, str):
                    if expiry_str.endswith('Z'):
                        expires_at = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                    elif '+' in expiry_str:
                        expires_at = datetime.fromisoformat(expiry_str)
                    else:
                        expires_at = datetime.fromisoformat(expiry_str).replace(tzinfo=timezone.utc)
            except ValueError:
                log.warning(f"无法解析过期时间: {expiry_str}")
        
        return cls(
            access_token=data.get('token') or data.get('access_token', ''),
            refresh_token=data.get('refresh_token'),
            client_id=data.get('client_id'),
            client_secret=data.get('client_secret'),
            expires_at=expires_at,
            project_id=data.get('project_id')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        result = {
            'access_token': self.access_token,
            'refresh_token': self.refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'project_id': self.project_id
        }
        
        if self.expires_at:
            result['expiry'] = self.expires_at.isoformat()
        
        return result


class Flow:
    """OAuth流程类"""
    
    def __init__(self, client_id: str, client_secret: str, scopes: List[str],
                 redirect_uri: str = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = scopes
        self.redirect_uri = redirect_uri
        
        # 反代配置将在使用时异步获取
        self.oauth_base_url = None
        self.token_endpoint = None
        self.auth_endpoint = "https://accounts.google.com/o/oauth2/auth"
        
        self.credentials: Optional[Credentials] = None
    
    def get_auth_url(self, state: str = None, **kwargs) -> str:
        """生成授权URL"""
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(self.scopes),
            'response_type': 'code',
            'access_type': 'offline',
            'prompt': 'consent',
            'include_granted_scopes': 'true'
        }
        
        if state:
            params['state'] = state
        
        params.update(kwargs)
        return f"{self.auth_endpoint}?{urlencode(params)}"
    
    async def exchange_code(self, code: str) -> Credentials:
        """用授权码换取token"""
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self.redirect_uri,
            'code': code,
            'grant_type': 'authorization_code'
        }
        
        try:
            oauth_base_url = await get_oauth_proxy_url()
            token_url = f"{oauth_base_url.rstrip('/')}/token"
            response = await post_async(
                token_url,
                data=data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            response.raise_for_status()
            
            token_data = response.json()
            
            # 计算过期时间
            expires_at = None
            if 'expires_in' in token_data:
                expires_in = int(token_data['expires_in'])
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            
            # 创建凭证对象
            self.credentials = Credentials(
                access_token=token_data['access_token'],
                refresh_token=token_data.get('refresh_token'),
                client_id=self.client_id,
                client_secret=self.client_secret,
                expires_at=expires_at
            )
            
            return self.credentials
            
        except Exception as e:
            error_msg = f"获取token失败: {str(e)}"
            log.error(error_msg)
            raise TokenError(error_msg)


class ServiceAccount:
    """Service Account类"""
    
    def __init__(self, email: str, private_key: str, project_id: str = None,
                 scopes: List[str] = None):
        self.email = email
        self.private_key = private_key
        self.project_id = project_id
        self.scopes = scopes or []
        
        # 反代配置将在使用时异步获取
        self.oauth_base_url = None
        self.token_endpoint = None
        
        self.access_token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """检查token是否过期"""
        if not self.expires_at:
            return True
        
        buffer = timedelta(minutes=3)
        return (self.expires_at - buffer) <= datetime.now(timezone.utc)
    
    def create_jwt(self) -> str:
        """创建JWT令牌"""
        now = int(time.time())
        
        payload = {
            'iss': self.email,
            'scope': ' '.join(self.scopes) if self.scopes else '',
            'aud': self.token_endpoint,
            'exp': now + 3600,
            'iat': now
        }
        
        return jwt.encode(payload, self.private_key, algorithm='RS256')
    
    async def get_access_token(self) -> str:
        """获取访问令牌"""
        if not self.is_expired() and self.access_token:
            return self.access_token
        
        assertion = self.create_jwt()
        
        data = {
            'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
            'assertion': assertion
        }
        
        try:
            oauth_base_url = await get_oauth_proxy_url()
            token_url = f"{oauth_base_url.rstrip('/')}/token"
            response = await post_async(
                token_url,
                data=data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            
            if 'expires_in' in token_data:
                expires_in = int(token_data['expires_in'])
                self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            
            return self.access_token
            
        except Exception as e:
            error_msg = f"Service Account获取token失败: {str(e)}"
            log.error(error_msg)
            raise TokenError(error_msg)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], scopes: List[str] = None) -> 'ServiceAccount':
        """从字典创建Service Account凭证"""
        return cls(
            email=data['client_email'],
            private_key=data['private_key'],
            project_id=data.get('project_id'),
            scopes=scopes
        )


# 工具函数
async def get_user_info(credentials: Credentials) -> Optional[Dict[str, Any]]:
    """获取用户信息"""
    await credentials.refresh_if_needed()
    
    try:
        googleapis_base_url = await get_googleapis_proxy_url()
        userinfo_url = f"{googleapis_base_url.rstrip('/')}/oauth2/v2/userinfo"
        response = await get_async(
            userinfo_url,
            headers={'Authorization': f'Bearer {credentials.access_token}'}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        log.error(f"获取用户信息失败: {e}")
        return None


async def get_user_email(credentials: Credentials) -> Optional[str]:
    """获取用户邮箱地址"""
    try:
        # 确保凭证有效
        await credentials.refresh_if_needed()
        
        # 调用Google userinfo API获取邮箱
        user_info = await get_user_info(credentials)
        if user_info:
            email = user_info.get("email")
            if email:
                log.info(f"成功获取邮箱地址: {email}")
                return email
            else:
                log.warning(f"userinfo响应中没有邮箱信息: {user_info}")
                return None
        else:
            log.warning("获取用户信息失败")
            return None
                
    except Exception as e:
        log.error(f"获取用户邮箱失败: {e}")
        return None


async def fetch_user_email_from_file(cred_data: Dict[str, Any]) -> Optional[str]:
    """从凭证数据获取用户邮箱地址（支持统一存储）"""
    try:
        # 直接从凭证数据创建凭证对象
        credentials = Credentials.from_dict(cred_data)
        if not credentials or not credentials.access_token:
            log.warning(f"无法从凭证数据创建凭证对象或获取访问令牌")
            return None
        
        # 获取邮箱
        return await get_user_email(credentials)
                
    except Exception as e:
        log.error(f"从凭证数据获取用户邮箱失败: {e}")
        return None


async def validate_token(token: str) -> Optional[Dict[str, Any]]:
    """验证访问令牌"""
    try:
        oauth_base_url = await get_oauth_proxy_url()
        tokeninfo_url = f"{oauth_base_url.rstrip('/')}/tokeninfo?access_token={token}"
        
        response = await get_async(tokeninfo_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        log.error(f"验证令牌失败: {e}")
        return None


async def enable_required_apis(credentials: Credentials, project_id: str) -> bool:
    """自动启用必需的API服务"""
    try:
        # 确保凭证有效
        if credentials.is_expired() and credentials.refresh_token:
            await credentials.refresh()
        
        headers = {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "geminicli-oauth/1.0",
        }
        
        # 需要启用的服务列表
        required_services = [
            "geminicloudassist.googleapis.com",  # Gemini Cloud Assist API
            "cloudaicompanion.googleapis.com"    # Gemini for Google Cloud API
        ]
        
        for service in required_services:
            log.info(f"正在检查并启用服务: {service}")
            
            # 检查服务是否已启用
            service_usage_base_url = await get_service_usage_api_url()
            check_url = f"{service_usage_base_url.rstrip('/')}/v1/projects/{project_id}/services/{service}"
            try:
                check_response = await get_async(check_url, headers=headers)
                if check_response.status_code == 200:
                    service_data = check_response.json()
                    if service_data.get("state") == "ENABLED":
                        log.info(f"服务 {service} 已启用")
                        continue
            except Exception as e:
                log.debug(f"检查服务状态失败，将尝试启用: {e}")
            
            # 启用服务
            enable_url = f"{service_usage_base_url.rstrip('/')}/v1/projects/{project_id}/services/{service}:enable"
            try:
                enable_response = await post_async(enable_url, headers=headers, json={})
                
                if enable_response.status_code in [200, 201]:
                    log.info(f"✅ 成功启用服务: {service}")
                elif enable_response.status_code == 400:
                    error_data = enable_response.json()
                    if "already enabled" in error_data.get("error", {}).get("message", "").lower():
                        log.info(f"✅ 服务 {service} 已经启用")
                    else:
                        log.warning(f"⚠️ 启用服务 {service} 时出现警告: {error_data}")
                else:
                    log.warning(f"⚠️ 启用服务 {service} 失败: {enable_response.status_code} - {enable_response.text}")
                    
            except Exception as e:
                log.warning(f"⚠️ 启用服务 {service} 时发生异常: {e}")
                
        return True
        
    except Exception as e:
        log.error(f"启用API服务时发生错误: {e}")
        return False


async def get_user_projects(credentials: Credentials) -> List[Dict[str, Any]]:
    """获取用户可访问的Google Cloud项目列表"""
    try:
        # 确保凭证有效
        if credentials.is_expired() and credentials.refresh_token:
            await credentials.refresh()
        
        headers = {
            "Authorization": f"Bearer {credentials.access_token}",
            "User-Agent": "geminicli-oauth/1.0",
        }
        
        # 使用Resource Manager API的正确域名和端点
        resource_manager_base_url = await get_resource_manager_api_url()
        url = f"{resource_manager_base_url.rstrip('/')}/v1/projects"
        log.info(f"正在调用API: {url}")
        response = await get_async(url, headers=headers)
        
        log.info(f"API响应状态码: {response.status_code}")
        if response.status_code != 200:
            log.error(f"API响应内容: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            projects = data.get('projects', [])
            # 只返回活跃的项目
            active_projects = [
                project for project in projects 
                if project.get('lifecycleState') == 'ACTIVE'
            ]
            log.info(f"获取到 {len(active_projects)} 个活跃项目")
            return active_projects
        else:
            log.warning(f"获取项目列表失败: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        log.error(f"获取用户项目列表失败: {e}")
        return []




async def select_default_project(projects: List[Dict[str, Any]]) -> Optional[str]:
    """从项目列表中选择默认项目"""
    if not projects:
        return None
    
    # 策略1：查找显示名称或项目ID包含"default"的项目
    for project in projects:
        display_name = project.get('displayName', '').lower()
        project_id = project.get('projectId', '')
        if 'default' in display_name or 'default' in project_id.lower():
            log.info(f"选择默认项目: {project_id} ({project.get('displayName', project_id)})")
            return project_id
    
    # 策略2：选择第一个项目
    first_project = projects[0]
    project_id = first_project.get('projectId', '')
    log.info(f"选择第一个项目作为默认: {project_id} ({first_project.get('displayName', project_id)})")
    return project_id


async def poll_operation_status(
    credentials: Credentials,
    operation_name: str,
    max_attempts: int = 60,
    poll_interval: float = 3.0
) -> bool:
    """
    轮询长时间运行操作的状态
    
    Args:
        credentials: OAuth凭证
        operation_name: 操作名称（从创建响应中获取）
        max_attempts: 最大轮询次数（默认60次，共2分钟）
        poll_interval: 每次轮询间隔秒数（默认2秒）
    
    Returns:
        True 如果操作成功完成，False 如果失败或超时
    """
    try:
        # 确保凭证有效
        if credentials.is_expired() and credentials.refresh_token:
            await credentials.refresh()
        
        headers = {
            "Authorization": f"Bearer {credentials.access_token}",
            "User-Agent": "geminicli-oauth/1.0",
        }
        
        resource_manager_base_url = await get_resource_manager_api_url()
        # 操作名称格式: operations/{operation-id}
        operation_url = f"{resource_manager_base_url.rstrip('/')}/v1/{operation_name}"
        
        log.info(f"开始轮询操作状态: {operation_name}")
        
        for attempt in range(max_attempts):
            try:
                response = await get_async(operation_url, headers=headers)
                
                if response.status_code == 200:
                    operation_data = response.json()
                    
                    # 检查操作是否完成
                    if operation_data.get('done', False):
                        # 检查是否有错误
                        if 'error' in operation_data:
                            error = operation_data['error']
                            log.error(f"❌ 操作失败: {error.get('message', error)}")
                            return False
                        
                        # 操作成功完成
                        log.info(f"✅ 操作成功完成 (耗时: {(attempt + 1) * poll_interval:.1f}秒)")
                        return True
                    else:
                        # 操作仍在进行中
                        if attempt % 5 == 0:  # 每10秒记录一次日志
                            log.debug(f"操作进行中... (已等待 {(attempt + 1) * poll_interval:.1f}秒)")
                        
                        # 等待后再次轮询
                        await asyncio.sleep(poll_interval)
                else:
                    log.warning(f"查询操作状态失败: {response.status_code} - {response.text}")
                    await asyncio.sleep(poll_interval)
                    
            except Exception as e:
                log.warning(f"轮询操作状态时出错 (尝试 {attempt + 1}/{max_attempts}): {e}")
                await asyncio.sleep(poll_interval)
        
        # 超时
        log.error(f"⏱️ 操作轮询超时 (超过 {max_attempts * poll_interval:.1f}秒)")
        return False
        
    except Exception as e:
        log.error(f"❌ 轮询操作状态时发生异常: {e}")
        return False


async def get_project_status(credentials: Credentials, project_id: str) -> Optional[Dict[str, Any]]:
    """
    获取项目的当前状态
    
    Args:
        credentials: OAuth凭证
        project_id: 项目ID
    
    Returns:
        项目信息字典，如果项目不存在或出错则返回None
    """
    try:
        # 确保凭证有效
        if credentials.is_expired() and credentials.refresh_token:
            await credentials.refresh()
        
        headers = {
            "Authorization": f"Bearer {credentials.access_token}",
            "User-Agent": "geminicli-oauth/1.0",
        }
        
        resource_manager_base_url = await get_resource_manager_api_url()
        project_url = f"{resource_manager_base_url.rstrip('/')}/v1/projects/{project_id}"
        
        response = await get_async(project_url, headers=headers)
        
        if response.status_code == 200:
            project_data = response.json()
            log.debug(f"项目 {project_id} 状态: {project_data.get('lifecycleState')}")
            return project_data
        else:
            log.debug(f"无法获取项目状态: {response.status_code}")
            return None
            
    except Exception as e:
        log.debug(f"获取项目状态时出错: {e}")
        return None


async def wait_for_project_ready(
    credentials: Credentials,
    project_id: str,
    max_attempts: int = 30,
    poll_interval: float = 2.0
) -> bool:
    """
    等待项目状态变为 ACTIVE
    
    Args:
        credentials: OAuth凭证
        project_id: 项目ID
        max_attempts: 最大轮询次数
        poll_interval: 轮询间隔秒数
    
    Returns:
        True 如果项目变为 ACTIVE，False 如果超时或失败
    """
    log.info(f"等待项目 {project_id} 状态变为 ACTIVE...")
    
    for attempt in range(max_attempts):
        project_data = await get_project_status(credentials, project_id)
        
        if project_data:
            lifecycle_state = project_data.get('lifecycleState')
            
            if lifecycle_state == 'ACTIVE':
                log.info(f"✅ 项目已就绪 (耗时: {(attempt + 1) * poll_interval:.1f}秒)")
                return True
            elif lifecycle_state in ['DELETE_REQUESTED', 'DELETE_IN_PROGRESS']:
                log.error(f"❌ 项目正在被删除")
                return False
            else:
                if attempt % 5 == 0:
                    log.debug(f"项目状态: {lifecycle_state}, 继续等待...")
        
        await asyncio.sleep(poll_interval)
    
    log.warning(f"⏱️ 等待项目就绪超时 (超过 {max_attempts * poll_interval:.1f}秒)")
    return False


async def create_google_cloud_project(credentials: Credentials, project_name: str = None) -> Optional[Dict[str, Any]]:
    """
    自动创建Google Cloud项目，并等待项目完全就绪
    
    Args:
        credentials: OAuth凭证
        project_name: 项目名称（可选，默认使用时间戳生成）
    
    Returns:
        创建的项目信息字典，失败返回None
    """
    try:
        # 确保凭证有效
        if credentials.is_expired() and credentials.refresh_token:
            await credentials.refresh()
        
        # 生成唯一的项目ID
        import random
        import string
        timestamp = int(time.time())
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        project_id = f"gemini-api-{timestamp}-{random_suffix}"
        
        # 如果没有指定项目名称，使用默认名称
        if not project_name:
            project_name = f"Gemini API Project {timestamp}"
        
        headers = {
            "Authorization": f"Bearer {credentials.access_token}",
            "Content-Type": "application/json",
            "User-Agent": "geminicli-oauth/1.0",
        }
        
        # 创建项目的请求体
        project_data = {
            "projectId": project_id,
            "name": project_name,
            "labels": {
                "created-by": "gcli2api",
                "auto-created": "true"
            }
        }
        
        # 调用Resource Manager API创建项目
        resource_manager_base_url = await get_resource_manager_api_url()
        create_url = f"{resource_manager_base_url.rstrip('/')}/v1/projects"
        
        log.info(f"正在创建Google Cloud项目: {project_id}")
        log.info(f"项目名称: {project_name}")
        log.info(f"API调用: POST {create_url}")
        
        response = await post_async(create_url, headers=headers, json=project_data)
        
        log.info(f"创建项目API响应状态码: {response.status_code}")
        
        if response.status_code in [200, 201]:
            created_project = response.json()
            log.info(f"✅ 项目创建请求已提交: {project_id}")
            
            # 检查响应中是否包含操作信息（长时间运行操作）
            if 'name' in created_project and created_project['name'].startswith('operations/'):
                # 这是一个长时间运行的操作
                operation_name = created_project['name']
                log.info(f"检测到长时间运行操作: {operation_name}")
                
                # 轮询操作状态
                operation_success = await poll_operation_status(
                    credentials, 
                    operation_name,
                    max_attempts=60,  # 最多3分钟
                    poll_interval=3.0
                )
                
                if not operation_success:
                    log.error("❌ 项目创建操作未能成功完成")
                    return None
            else:
                # 同步响应，项目已创建
                log.info("项目创建请求已同步完成")
            
            # 等待项目状态变为 ACTIVE
            project_ready = await wait_for_project_ready(
                credentials,
                project_id,
                max_attempts=30,  # 最多1分钟
                poll_interval=2.0
            )
            
            if not project_ready:
                log.warning("⚠️ 项目创建完成但未能确认其状态为ACTIVE，继续尝试...")
            
            # 获取最终的项目信息
            final_project_data = await get_project_status(credentials, project_id)
            
            if final_project_data:
                log.info(f"项目详情: lifecycleState={final_project_data.get('lifecycleState')}, createTime={final_project_data.get('createTime')}")
            
            # 尝试启用必需的API
            log.info("正在为新项目启用必需的API服务...")
            await enable_required_apis(credentials, project_id)
            
            return {
                'projectId': project_id,
                'name': project_name,
                'displayName': project_name,
                'lifecycleState': final_project_data.get('lifecycleState', 'ACTIVE') if final_project_data else 'ACTIVE',
                'createTime': final_project_data.get('createTime') if final_project_data else created_project.get('createTime'),
                'auto_created': True
            }
        else:
            error_text = response.text
            log.error(f"❌ 创建项目失败: {response.status_code} - {error_text}")
            
            # 解析错误信息
            try:
                error_data = response.json()
                error_message = error_data.get('error', {}).get('message', error_text)
                log.error(f"错误详情: {error_message}")
                
                # 检查是否是配额限制错误
                if 'quota' in error_message.lower() or 'limit' in error_message.lower():
                    log.error("⚠️ 可能已达到项目创建配额限制（通常为12个项目）")
                    log.error("建议：删除不用的项目或申请增加配额")
            except:
                pass
            
            return None
            
    except Exception as e:
        log.error(f"❌ 创建Google Cloud项目时发生异常: {e}")
        import traceback
        log.error(traceback.format_exc())
        return None