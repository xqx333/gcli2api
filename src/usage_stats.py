"""
Usage statistics module for tracking API calls per credential file.
Uses the simpler logic: compare current time with next_reset_time.
"""
import asyncio
import os
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

from log import log
from .storage_adapter import get_storage_adapter


def _get_next_utc_7am() -> datetime:
    """Calculate the next UTC 07:00 time for quota reset."""
    now = datetime.now(timezone.utc)
    today_7am = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if now < today_7am:
        return today_7am
    return today_7am + timedelta(days=1)


def _normalize_filename(filename: str) -> str:
    """Normalize filename to relative path for consistent storage."""
    if not filename:
        return ""
    if os.path.sep not in filename and "/" not in filename:
        return filename
    return os.path.basename(filename)


def _is_gemini_2_5_pro(model_name: str) -> bool:
    """Check if model is gemini-2.5-pro variant (including prefixes and suffixes)."""
    if not model_name:
        return False

    try:
        from config import get_base_model_name, get_base_model_from_feature_model

        base_with_suffix = get_base_model_from_feature_model(model_name)
        pure_base_model = get_base_model_name(base_with_suffix)
        return pure_base_model == "gemini-2.5-pro"
    except ImportError:
        clean_model = model_name
        for prefix in ["流式抗截断", "假流式"]:
            if clean_model.startswith(prefix):
                clean_model = clean_model[len(prefix):]
                break

        for suffix in ["-maxthinking", "-nothinking", "-search"]:
            if clean_model.endswith(suffix):
                clean_model = clean_model[:-len(suffix)]
                break

        return clean_model == "gemini-2.5-pro"


class UsageStats:
    """Lightweight usage statistics manager that loads and persists per file on demand."""

    _STATS_FIELDS = {
        "gemini_2_5_pro_calls",
        "total_calls",
        "next_reset_time",
        "daily_limit_gemini_2_5_pro",
        "daily_limit_total",
    }

    def __init__(self):
        self._lock = asyncio.Lock()
        self._storage_adapter = None
        self._initialized = False
        self._stats_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_capacity = 200

    async def initialize(self):
        if self._initialized:
            return
        self._storage_adapter = await get_storage_adapter()
        self._initialized = True
        log.debug("Usage statistics module initialized")

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.initialize()

    def _default_stats(self) -> Dict[str, Any]:
        next_reset = _get_next_utc_7am().isoformat()
        return {
            "gemini_2_5_pro_calls": 0,
            "total_calls": 0,
            "daily_limit_gemini_2_5_pro": 100,
            "daily_limit_total": 1000,
            "next_reset_time": next_reset,
        }

    def _sanitize_stats(self, stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(stats, dict):
            return self._default_stats()

        baseline = self._default_stats()
        sanitized = baseline.copy()
        for key in self._STATS_FIELDS:
            if key in stats and stats[key] is not None:
                sanitized[key] = stats[key]
        if not sanitized.get("next_reset_time"):
            sanitized["next_reset_time"] = baseline["next_reset_time"]
        return sanitized

    def _stats_payload(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "gemini_2_5_pro_calls": stats.get("gemini_2_5_pro_calls", 0),
            "total_calls": stats.get("total_calls", 0),
            "next_reset_time": stats.get("next_reset_time"),
            "daily_limit_gemini_2_5_pro": stats.get("daily_limit_gemini_2_5_pro", 100),
            "daily_limit_total": stats.get("daily_limit_total", 1000),
        }

    def _stats_response(self, filename: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "filename": filename,
            "gemini_2_5_pro_calls": stats.get("gemini_2_5_pro_calls", 0),
            "total_calls": stats.get("total_calls", 0),
            "daily_limit_gemini_2_5_pro": stats.get("daily_limit_gemini_2_5_pro", 100),
            "daily_limit_total": stats.get("daily_limit_total", 1000),
            "next_reset_time": stats.get("next_reset_time"),
        }

    async def _get_or_load_stats(self, filename: str) -> Dict[str, Any]:
        normalized = _normalize_filename(filename)
        async with self._lock:
            cached = self._stats_cache.get(normalized)
            if cached is not None:
                self._stats_cache.move_to_end(normalized)
                return cached
        fetched = await self._storage_adapter.get_usage_stats(normalized)
        sanitized = self._sanitize_stats(fetched)
        async with self._lock:
            cached = self._stats_cache.get(normalized)
            if cached is not None:
                cached.clear()
                cached.update(sanitized)
                self._stats_cache.move_to_end(normalized)
                return cached
            self._stats_cache[normalized] = sanitized
            self._stats_cache.move_to_end(normalized)
            while len(self._stats_cache) > self._cache_capacity:
                self._stats_cache.popitem(last=False)
        return sanitized

    def _check_and_reset_daily_quota(self, stats: Dict[str, Any]) -> bool:
        try:
            next_reset_str = stats.get("next_reset_time")
            if not next_reset_str:
                stats["next_reset_time"] = _get_next_utc_7am().isoformat()
                return False

            next_reset = datetime.fromisoformat(next_reset_str)
            now = datetime.now(timezone.utc)

            if now >= next_reset:
                new_next_reset = _get_next_utc_7am()
                stats.update({
                    "gemini_2_5_pro_calls": 0,
                    "total_calls": 0,
                    "next_reset_time": new_next_reset.isoformat(),
                })
                return True
            return False
        except Exception as exc:
            log.error(f"Error in daily quota reset check: {exc}")
            return False

    async def record_successful_call(self, filename: str, model_name: str):
        await self._ensure_initialized()
        normalized = _normalize_filename(filename)
        stats = await self._get_or_load_stats(normalized)

        async with self._lock:
            reset_performed = self._check_and_reset_daily_quota(stats)
            stats["total_calls"] += 1
            if _is_gemini_2_5_pro(model_name):
                stats["gemini_2_5_pro_calls"] += 1
            payload = self._stats_payload(stats)

        await self._storage_adapter.update_usage_stats(normalized, payload)
        log.debug(
            "Usage recorded - File: %s, Model: %s, Gemini 2.5 Pro: %s/%s, Total: %s/%s",
            normalized,
            model_name,
            stats.get("gemini_2_5_pro_calls"),
            stats.get("daily_limit_gemini_2_5_pro", 100),
            stats.get("total_calls"),
            stats.get("daily_limit_total", 1000),
        )
        if reset_performed:
            log.info(f"Daily quota was reset for {normalized}")

    async def get_usage_stats(self, filename: str = None) -> Dict[str, Any]:
        await self._ensure_initialized()

        if filename:
            normalized = _normalize_filename(filename)
            stats = await self._get_or_load_stats(normalized)
            async with self._lock:
                reset_performed = self._check_and_reset_daily_quota(stats)
                response = self._stats_response(normalized, stats)
                payload = self._stats_payload(stats) if reset_performed else None
            if payload:
                await self._storage_adapter.update_usage_stats(normalized, payload)
            return response

        all_stats = await self._storage_adapter.get_all_usage_stats()
        result: Dict[str, Any] = {}
        updates: Dict[str, Dict[str, Any]] = {}

        async with self._lock:
            for raw_name, raw_stats in all_stats.items():
                normalized = _normalize_filename(raw_name)
                sanitized = self._sanitize_stats(raw_stats)
                if self._check_and_reset_daily_quota(sanitized):
                    updates[normalized] = self._stats_payload(sanitized)
                result[normalized] = self._stats_response(normalized, sanitized)

                cached = self._stats_cache.get(normalized)
                if cached is None:
                    self._stats_cache[normalized] = sanitized
                else:
                    cached.clear()
                    cached.update(sanitized)
                self._stats_cache.move_to_end(normalized)

            while len(self._stats_cache) > self._cache_capacity:
                self._stats_cache.popitem(last=False)

        for normalized, payload in updates.items():
            await self._storage_adapter.update_usage_stats(normalized, payload)

        return result

    async def get_aggregated_stats(self) -> Dict[str, Any]:
        all_stats = await self.get_usage_stats()
        total_gemini_2_5_pro = 0
        total_all_models = 0
        total_files = len(all_stats)

        for stats in all_stats.values():
            total_gemini_2_5_pro += stats["gemini_2_5_pro_calls"]
            total_all_models += stats["total_calls"]

        return {
            "total_files": total_files,
            "total_gemini_2_5_pro_calls": total_gemini_2_5_pro,
            "total_all_model_calls": total_all_models,
            "avg_gemini_2_5_pro_per_file": total_gemini_2_5_pro / max(total_files, 1),
            "avg_total_per_file": total_all_models / max(total_files, 1),
            "next_reset_time": _get_next_utc_7am().isoformat(),
        }

    async def update_daily_limits(
        self,
        filename: str,
        gemini_2_5_pro_limit: int = None,
        total_limit: int = None,
    ):
        await self._ensure_initialized()
        normalized = _normalize_filename(filename)
        stats = await self._get_or_load_stats(normalized)

        async with self._lock:
            if gemini_2_5_pro_limit is not None:
                stats["daily_limit_gemini_2_5_pro"] = gemini_2_5_pro_limit
            if total_limit is not None:
                stats["daily_limit_total"] = total_limit
            payload = self._stats_payload(stats)

        await self._storage_adapter.update_usage_stats(normalized, payload)
        log.info(
            "Updated daily limits for %s: Gemini 2.5 Pro=%s, Total=%s",
            normalized,
            stats.get("daily_limit_gemini_2_5_pro", 100),
            stats.get("daily_limit_total", 1000),
        )

    async def reset_stats(self, filename: str = None):
        await self._ensure_initialized()

        if filename:
            normalized = _normalize_filename(filename)
            default_stats = self._default_stats()
            stats = await self._get_or_load_stats(normalized)
            async with self._lock:
                stats.clear()
                stats.update(default_stats)
                payload = self._stats_payload(stats)
            await self._storage_adapter.update_usage_stats(normalized, payload)
            log.info(f"Reset usage statistics for {normalized}")
            return

        credential_names = await self._storage_adapter.list_credentials()

        for name in credential_names:
            normalized = _normalize_filename(name)
            default_stats = self._default_stats()
            payload = self._stats_payload(default_stats)
            await self._storage_adapter.update_usage_stats(normalized, payload)
            async with self._lock:
                self._stats_cache[normalized] = default_stats
                self._stats_cache.move_to_end(normalized)
                while len(self._stats_cache) > self._cache_capacity:
                    self._stats_cache.popitem(last=False)

        async with self._lock:
            while len(self._stats_cache) > self._cache_capacity:
                self._stats_cache.popitem(last=False)

        log.info("Reset usage statistics for all credential files")


_usage_stats_instance: Optional[UsageStats] = None


async def get_usage_stats_instance() -> UsageStats:
    """Get the global usage statistics instance."""
    global _usage_stats_instance
    if _usage_stats_instance is None:
        _usage_stats_instance = UsageStats()
        await _usage_stats_instance.initialize()
    return _usage_stats_instance


async def record_successful_call(filename: str, model_name: str):
    """Convenience function to record a successful API call."""
    stats = await get_usage_stats_instance()
    await stats.record_successful_call(filename, model_name)


async def get_usage_stats(filename: str = None) -> Dict[str, Any]:
    """Convenience function to get usage statistics."""
    stats = await get_usage_stats_instance()
    return await stats.get_usage_stats(filename)


async def get_aggregated_stats() -> Dict[str, Any]:
    """Convenience function to get aggregated statistics."""
    stats = await get_usage_stats_instance()
    return await stats.get_aggregated_stats()
