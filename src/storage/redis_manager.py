"""
Redis数据库管理器，使用哈希表设计和统一缓存。
所有凭证数据存储在一个哈希表中，配置数据存储在另一个哈希表中。
"""
import asyncio
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

import redis.asyncio as redis
from log import log
from .cache_manager import UnifiedCacheManager, CacheBackend


STATE_FIELDS = {
    "error_codes", "error_details", "disabled", "last_success", "user_email"
}

STATS_FIELDS = {
    "gemini_2_5_pro_calls",
    "total_calls",
    "next_reset_time",
    "daily_limit_gemini_2_5_pro",
    "daily_limit_total"
}

class RedisCacheBackend(CacheBackend):
    """Redis缓存后端实现"""
    
    def __init__(self, redis_client: redis.Redis, hash_name: str):
        self._client = redis_client
        self._hash_name = hash_name
    
    async def load_data(self) -> Dict[str, Any]:
        """从Redis哈希表加载数据"""
        try:
            hash_data = await self._client.hgetall(self._hash_name)
            if not hash_data:
                return {}
            
            result = {}
            for key, value_str in hash_data.items():
                try:
                    result[key] = json.loads(value_str)
                except json.JSONDecodeError as e:
                    log.error(f"Error deserializing Redis data for key {key}: {e}")
                    continue
            return result
        except Exception as e:
            log.error(f"Error loading data from Redis hash {self._hash_name}: {e}")
            return {}
    
    async def write_data(self, data: Dict[str, Any]) -> bool:
        """将数据写入Redis哈希表"""
        try:
            if not data:
                await self._client.delete(self._hash_name)
                return True
            
            hash_data = {}
            for key, value in data.items():
                try:
                    hash_data[key] = json.dumps(value, ensure_ascii=False)
                except (TypeError, ValueError) as e:
                    log.error(f"Error serializing data for key {key}: {e}")
                    continue
            
            if not hash_data:
                return True
            
            pipe = self._client.pipeline()
            pipe.delete(self._hash_name)
            pipe.hset(self._hash_name, mapping=hash_data)
            await pipe.execute()
            return True
        except Exception as e:
            log.error(f"Error writing data to Redis hash {self._hash_name}: {e}")
            return False


class RedisManager:
    """Redis数据库管理器"""
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._initialized = False
        self._lock = asyncio.Lock()
        
        # 配置
        self._connection_uri = None
        self._database_index = 0
        
        # 哈希表设计 - 所有凭证存在一个哈希表中
        self._credentials_hash_name = "gcli2api:credentials"
        self._config_hash_name = "gcli2api:config"
        
        # 性能监控
        self._operation_count = 0
        self._operation_times = deque(maxlen=5000)
        
        # 统一缓存管理器
        self._credentials_cache_manager: Optional[UnifiedCacheManager] = None
        self._config_cache_manager: Optional[UnifiedCacheManager] = None
        
        # 写入配置参数
        self._write_delay = 1.0  # 写入延迟（秒）
        self._cache_ttl = 300  # 缓存TTL（秒）
    
    async def initialize(self):
        """初始化Redis连接"""
        async with self._lock:
            if self._initialized:
                return
                
            try:
                # 获取连接配置
                self._connection_uri = os.getenv("REDIS_URI", "redis://localhost:6379")
                self._database_index = int(os.getenv("REDIS_DATABASE", "0"))
                
                # 建立连接 - 使用最简配置确保兼容性
                # 检查是否需要 SSL
                if self._connection_uri.startswith("rediss://"):
                    # SSL 连接
                    self._client = redis.from_url(
                        self._connection_uri,
                        db=self._database_index,
                        decode_responses=True,
                        ssl_cert_reqs=None,
                        ssl_check_hostname=False,
                        ssl_ca_certs=None
                    )
                else:
                    # 普通连接
                    self._client = redis.from_url(
                        self._connection_uri,
                        db=self._database_index,
                        decode_responses=True
                    )
                
                # 验证连接
                await self._client.ping()
                
                # 创建缓存管理器
                credentials_backend = RedisCacheBackend(self._client, self._credentials_hash_name)
                config_backend = RedisCacheBackend(self._client, self._config_hash_name)
                
                self._credentials_cache_manager = UnifiedCacheManager(
                    credentials_backend, 
                    cache_ttl=self._cache_ttl, 
                    write_delay=self._write_delay,
                    name="credentials"
                )
                
                self._config_cache_manager = UnifiedCacheManager(
                    config_backend,
                    cache_ttl=self._cache_ttl,
                    write_delay=self._write_delay,
                    name="config"
                )
                
                # 启动缓存管理器
                await self._credentials_cache_manager.start()
                await self._config_cache_manager.start()
                
                self._initialized = True
                log.info(f"Redis connection established to database {self._database_index} with unified cache")
                
            except Exception as e:
                log.error(f"Error initializing Redis: {e}")
                raise
    
    async def close(self):
        """关闭Redis连接"""
        # 停止缓存管理器
        if self._credentials_cache_manager:
            await self._credentials_cache_manager.stop()
        if self._config_cache_manager:
            await self._config_cache_manager.stop()
        
        if self._client:
            await self._client.close()
            self._initialized = False
            log.info("Redis connection closed with unified cache flushed")
    
    def _ensure_initialized(self):
        """确保已初始化"""
        if not self._initialized:
            raise RuntimeError("Redis manager not initialized")
    
    def _get_default_state(self) -> Dict[str, Any]:
        """获取默认状态数据"""
        return {
            "error_codes": [],
            "error_details": {},
            "disabled": False,
            "last_success": time.time(),
            "user_email": None,
        }
    
    def _get_default_stats(self) -> Dict[str, Any]:
        """获取默认统计数据"""
        return {
            "gemini_2_5_pro_calls": 0,
            "total_calls": 0,
            "next_reset_time": None,
            "daily_limit_gemini_2_5_pro": 100,
            "daily_limit_total": 1000
        }
    
    # ============ 凭证管理 ============
    
    def _create_default_entry(self) -> Dict[str, Any]:
        return {
            "credential": {},
            "state": self._get_default_state(),
            "stats": self._get_default_stats()
        }

    def _normalize_entry(self, entry: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], bool]:
        changed = False

        default_state = self._get_default_state()
        default_stats = self._get_default_stats()
        normalized = {
            "credential": {},
            "state": default_state.copy(),
            "stats": default_stats.copy()
        }

        if not isinstance(entry, dict) or not entry:
            return normalized, True

        credential_data = entry.get("credential")
        if isinstance(credential_data, dict):
            normalized["credential"] = credential_data
        else:
            candidate_fields = {
                key: value
                for key, value in entry.items()
                if key not in ("credential", "state", "stats")
                and key not in STATE_FIELDS
                and key not in STATS_FIELDS
            }
            if candidate_fields:
                normalized["credential"] = candidate_fields
                changed = True
            elif not ({"credential", "state", "stats"} & set(entry.keys())):
                normalized["credential"] = dict(entry)
                changed = True

        original_state = entry.get("state") if isinstance(entry.get("state"), dict) else None
        if original_state:
            normalized["state"].update(original_state)
        else:
            changed = True

        for field in STATE_FIELDS:
            if field in entry and field not in normalized["state"]:
                normalized["state"][field] = entry[field]
                changed = True

        if original_state is not None:
            missing_state_fields = [field for field in default_state if field not in original_state]
            if missing_state_fields:
                for field in missing_state_fields:
                    normalized["state"].setdefault(field, default_state[field])
                changed = True
        else:
            for field, value in default_state.items():
                if field not in normalized["state"]:
                    normalized["state"][field] = value
            changed = True

        original_stats = entry.get("stats") if isinstance(entry.get("stats"), dict) else None
        if original_stats:
            normalized["stats"].update(original_stats)
        else:
            changed = True

        for field in STATS_FIELDS:
            if field in entry and field not in normalized["stats"]:
                normalized["stats"][field] = entry[field]
                changed = True

        if original_stats is not None:
            missing_stats_fields = [field for field in default_stats if field not in original_stats]
            if missing_stats_fields:
                for field in missing_stats_fields:
                    normalized["stats"].setdefault(field, default_stats[field])
                changed = True
        else:
            for field, value in default_stats.items():
                if field not in normalized["stats"]:
                    normalized["stats"][field] = value
            changed = True

        return normalized, changed

    async def store_credential(self, filename: str, credential_data: Dict[str, Any]) -> bool:
        """存储凭证数据到统一缓存"""
        self._ensure_initialized()
        start_time = time.time()

        try:
            existing_data = await self._credentials_cache_manager.get(filename)
            normalized_entry, _ = self._normalize_entry(existing_data)
            normalized_entry["credential"] = credential_data

            success = await self._credentials_cache_manager.set(filename, normalized_entry)

            self._operation_count += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)

            log.debug(f"Stored credential to unified cache: {filename} in {operation_time:.3f}s")
            return success

        except Exception as e:
            operation_time = time.time() - start_time
            log.error(f"Error storing credential {filename} in {operation_time:.3f}s: {e}")
            return False

    async def get_credential(self, filename: str) -> Optional[Dict[str, Any]]:
        """从统一缓存获取凭证数据"""
        self._ensure_initialized()
        start_time = time.time()

        try:
            credential_entry = await self._credentials_cache_manager.get(filename)

            if not credential_entry:
                operation_time = time.time() - start_time
                self._operation_count += 1
                self._operation_times.append(operation_time)
                return None

            normalized_entry, changed = self._normalize_entry(credential_entry)
            if changed:
                await self._credentials_cache_manager.set(filename, normalized_entry)

            operation_time = time.time() - start_time
            self._operation_count += 1
            self._operation_times.append(operation_time)

            log.debug(f"Retrieved credential from unified cache: {filename} in {operation_time:.3f}s")
            return normalized_entry.get("credential")

        except Exception as e:
            operation_time = time.time() - start_time
            log.error(f"Error getting credential {filename} in {operation_time:.3f}s: {e}")
            return None

    async def list_credentials(self) -> List[str]:
        """从统一缓存列出所有凭证文件名"""
        self._ensure_initialized()
        start_time = time.time()
        
        try:
            all_data = await self._credentials_cache_manager.get_all()
            filenames = list(all_data.keys())
            
            # 性能监控
            self._operation_count += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)
            
            log.debug(f"Listed {len(filenames)} credentials from unified cache in {operation_time:.3f}s")
            return filenames
            
        except Exception as e:
            operation_time = time.time() - start_time
            log.error(f"Error listing credentials in {operation_time:.3f}s: {e}")
            return []
    
    async def delete_credential(self, filename: str) -> bool:
        """从统一缓存删除凭证及所有相关数据"""
        self._ensure_initialized()
        start_time = time.time()
        
        try:
            success = await self._credentials_cache_manager.delete(filename)
            
            # 性能监控
            self._operation_count += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)
            
            log.debug(f"Deleted credential from unified cache: {filename} in {operation_time:.3f}s")
            return success
                
        except Exception as e:
            operation_time = time.time() - start_time
            log.error(f"Error deleting credential {filename} in {operation_time:.3f}s: {e}")
            return False
    
    # ============ 状态管理 ============
    
    async def update_credential_state(self, filename: str, state_updates: Dict[str, Any]) -> bool:
        """更新凭证状态（统一缓存）"""
        self._ensure_initialized()
        start_time = time.time()

        try:
            existing_data = await self._credentials_cache_manager.get(filename)
            if not existing_data:
                operation_time = time.time() - start_time
                log.debug(f"Skip state update for missing credential: {filename}")
                self._operation_count += 1
                self._operation_times.append(operation_time)
                return True

            has_credential = False
            if isinstance(existing_data, dict):
                credential_section = existing_data.get('credential')
                if isinstance(credential_section, dict) and credential_section:
                    has_credential = True
                else:
                    for key in existing_data.keys():
                        if key not in ('credential', 'state', 'stats') and key not in STATE_FIELDS and key not in STATS_FIELDS:
                            has_credential = True
                            break

            if not has_credential:
                operation_time = time.time() - start_time
                log.debug(f"Skip state update for credential without data: {filename}")
                self._operation_count += 1
                self._operation_times.append(operation_time)
                return True

            normalized_entry, _ = self._normalize_entry(existing_data)
            normalized_entry["state"].update(state_updates)

            success = await self._credentials_cache_manager.set(filename, normalized_entry)

            self._operation_count += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)

            log.debug(f"Updated credential state in unified cache: {filename} in {operation_time:.3f}s")
            return success

        except Exception as e:
            operation_time = time.time() - start_time
            log.error(f"Error updating credential state {filename} in {operation_time:.3f}s: {e}")
            return False
    async def get_credential_state(self, filename: str) -> Dict[str, Any]:
        """从统一缓存获取凭证状态"""
        self._ensure_initialized()
        start_time = time.time()

        try:
            credential_entry = await self._credentials_cache_manager.get(filename)

            if not credential_entry:
                operation_time = time.time() - start_time
                self._operation_count += 1
                self._operation_times.append(operation_time)
                return self._get_default_state()

            normalized_entry, changed = self._normalize_entry(credential_entry)
            if changed:
                await self._credentials_cache_manager.set(filename, normalized_entry)

            operation_time = time.time() - start_time
            self._operation_count += 1
            self._operation_times.append(operation_time)

            log.debug(f"Retrieved credential state from unified cache: {filename} in {operation_time:.3f}s")
            return normalized_entry["state"]

        except Exception as e:
            operation_time = time.time() - start_time
            log.error(f"Error getting credential state {filename} in {operation_time:.3f}s: {e}")
            return self._get_default_state()

    async def get_all_credential_states(self) -> Dict[str, Dict[str, Any]]:
        """从统一缓存获取所有凭证状态"""
        self._ensure_initialized()
        start_time = time.time()

        try:
            all_data = await self._credentials_cache_manager.get_all()
            states: Dict[str, Dict[str, Any]] = {}

            for filename, raw_entry in all_data.items():
                normalized_entry, changed = self._normalize_entry(raw_entry)
                if changed:
                    await self._credentials_cache_manager.set(filename, normalized_entry)
                states[filename] = normalized_entry["state"]

            self._operation_count += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)

            log.debug(f"Retrieved all credential states from unified cache ({len(states)}) in {operation_time:.3f}s")
            return states

        except Exception as e:
            operation_time = time.time() - start_time
            log.error(f"Error getting all credential states in {operation_time:.3f}s: {e}")
            return {}

    # ============ 配置管理 ============
    
    async def set_config(self, key: str, value: Any) -> bool:
        """设置配置到统一缓存"""
        self._ensure_initialized()
        start_time = time.time()
        
        try:
            success = await self._config_cache_manager.set(key, value)
            
            # 性能监控
            self._operation_count += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)
            
            log.debug(f"Set config to unified cache: {key} in {operation_time:.3f}s")
            return success
            
        except Exception as e:
            operation_time = time.time() - start_time
            log.error(f"Error setting config {key} in {operation_time:.3f}s: {e}")
            return False
    
    async def get_config(self, key: str, default: Any = None) -> Any:
        """从统一缓存获取配置"""
        self._ensure_initialized()
        return await self._config_cache_manager.get(key, default)
    
    async def get_all_config(self) -> Dict[str, Any]:
        """从统一缓存获取所有配置"""
        self._ensure_initialized()
        return await self._config_cache_manager.get_all()
    
    async def delete_config(self, key: str) -> bool:
        """从统一缓存删除配置"""
        self._ensure_initialized()
        return await self._config_cache_manager.delete(key)
    
    # ============ 使用统计管理 ============
    
    async def update_usage_stats(self, filename: str, stats_updates: Dict[str, Any]) -> bool:
        """更新使用统计（使用统一缓存）"""
        self._ensure_initialized()
        start_time = time.time()
        
        try:
            # 获取现有数据或创建新数据
            existing_data = await self._credentials_cache_manager.get(filename, {})
            
            if not existing_data:
                existing_data = {
                    "credential": {},
                    "state": self._get_default_state(),
                    "stats": self._get_default_stats()
                }
            
            # 更新统计数据
            existing_data["stats"].update(stats_updates)
            
            success = await self._credentials_cache_manager.set(filename, existing_data)
            
            # 性能监控
            self._operation_count += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)
            
            log.debug(f"Updated usage stats in unified cache: {filename} in {operation_time:.3f}s")
            return success
                
        except Exception as e:
            operation_time = time.time() - start_time
            log.error(f"Error updating usage stats {filename} in {operation_time:.3f}s: {e}")
            return False
    
    async def get_usage_stats(self, filename: str) -> Dict[str, Any]:
        """从统一缓存获取使用统计"""
        self._ensure_initialized()
        start_time = time.time()
        
        try:
            credential_entry = await self._credentials_cache_manager.get(filename)
            
            # 性能监控
            self._operation_count += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)
            
            if credential_entry and "stats" in credential_entry:
                log.debug(f"Retrieved usage stats from unified cache: {filename} in {operation_time:.3f}s")
                return credential_entry["stats"]
            else:
                return self._get_default_stats()
                
        except Exception as e:
            operation_time = time.time() - start_time
            log.error(f"Error getting usage stats {filename} in {operation_time:.3f}s: {e}")
            return self._get_default_stats()
    
    async def get_all_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all credentials from the unified cache."""
        self._ensure_initialized()
        start_time = time.time()

        try:
            all_data = await self._credentials_cache_manager.get_all()
            stats: Dict[str, Dict[str, Any]] = {}

            for filename, raw_entry in all_data.items():
                normalized_entry, changed = self._normalize_entry(raw_entry)
                if changed:
                    await self._credentials_cache_manager.set(filename, normalized_entry)
                stats[filename] = normalized_entry["stats"]

            self._operation_count += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)

            log.debug(f"Retrieved all usage stats from unified cache ({len(stats)}) in {operation_time:.3f}s")
            return stats

        except Exception as e:
            operation_time = time.time() - start_time
            log.error(f"Error getting all usage stats in {operation_time:.3f}s: {e}")
            return {}
