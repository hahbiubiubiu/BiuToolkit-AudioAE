# -*- coding: utf-8 -*-
"""Unified Tencent ASR entrypoint.

Only this dispatcher loads .env for multi-account fallback. Low-level API files do
not load .env at import time; they accept explicit credentials and only load a
local .env when called directly without credentials.
"""
from __future__ import annotations

import io
import logging
import os
import random
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from dotenv import load_dotenv

from .exception import (
    TencentASRError,
    TencentASRAuthError,
    TencentASRNetworkError,
    TencentASRResourceExhaustedError,
    TencentASRServerError,
    TencentASRTimeoutError,
    read_audio_file,
    validate_audio_bytes,
)
from . import recognition
from . import recognition_flash
from . import recognition_realtime
from . import recognition_sentence

logger = logging.getLogger(__name__)
InputKind = Literal["bytes", "file"]


@dataclass(frozen=True)
class AccountConfig:
    account_id: str
    secret_id: str
    secret_key: str
    appid: str | None = None


@dataclass(frozen=True)
class ASRService:
    name: str
    by_bytes: Callable[..., str]
    by_file: Callable[..., str]


SERVICES: tuple[ASRService, ...] = (
    ASRService("flash", recognition_flash.asr_by_bytes, recognition_flash.asr_by_file),
    ASRService("sentence", recognition_sentence.asr_by_bytes, recognition_sentence.asr_by_file),
    ASRService("realtime", recognition_realtime.asr_by_bytes, recognition_realtime.asr_by_file),
    ASRService("task", recognition.asr_by_bytes, recognition.asr_by_file),
)

class TencentASR:
    """Tencent ASR unified class entrypoint with multi-account fallback."""

    def __init__(
        self,
        services: Iterable[ASRService] | None = None,
        *,
        load_env: bool = True,
        secret_id: str | None = None,
        secret_key: str | None = None,
        appid: str | None = None,
    ):
        if (secret_id is None) ^ (secret_key is None):
            raise TencentASRAuthError(4002, "初始化 TencentASR 时 secret_id 与 secret_key 必须同时提供")

        self.services = tuple(services) if services is not None else SERVICES
        self._disabled_services: dict[str, set[str]] = {}
        self._disabled_accounts: set[str] = set()
        self._accounts_cache: list[AccountConfig] | None = None
        self._state_lock = threading.RLock()
        self._default_secret_id = secret_id
        self._default_secret_key = secret_key
        self._default_appid = appid
        self._load_env = load_env and not (secret_id and secret_key)

    def _load_dispatcher_env(self) -> None:
        """Load .env from the directory of this script."""
        if self._load_env:
            load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

    def load_account_configs(self, *, force_reload: bool = False) -> list[AccountConfig]:
        """Load Tencent account configs from local .env / process env."""
        with self._state_lock:
            if self._accounts_cache is not None and not force_reload:
                return list(self._accounts_cache)

            self._load_dispatcher_env()
            accounts: list[AccountConfig] = []
            i = 1
            while True:
                sid = os.getenv(f"ACCOUNT_SECRET_ID_{i}")
                skey = os.getenv(f"ACCOUNT_SECRET_KEY_{i}")
                appid = os.getenv(f"ACCOUNT_APP_ID_{i}")
                if not any([sid, skey, appid]):
                    break
                if sid and skey:
                    accounts.append(
                        AccountConfig(
                            f"account_{i}",
                            sid.strip(),
                            skey.strip(),
                            appid.strip() if appid else None,
                        )
                    )
                else:
                    logger.warning("Skip incomplete Tencent account_%s", i)
                i += 1

            if not accounts:
                sid = os.getenv("TENCENTCLOUD_SECRET_ID")
                skey = os.getenv("TENCENTCLOUD_SECRET_KEY")
                appid = os.getenv("APPID")
                if sid and skey:
                    accounts.append(
                        AccountConfig("default", sid.strip(), skey.strip(), appid.strip() if appid else None)
                    )

            if not accounts:
                raise TencentASRAuthError(
                    4002,
                    "未找到腾讯云账户配置：请配置 ACCOUNT_SECRET_ID_1/ACCOUNT_SECRET_KEY_1/ACCOUNT_APP_ID_1 "
                    "或 TENCENTCLOUD_SECRET_ID/TENCENTCLOUD_SECRET_KEY/APPID",
                )

            for account in accounts:
                self._disabled_services.setdefault(account.account_id, set())
            self._accounts_cache = accounts
            return list(accounts)

    def reset_fallback_state(self, *, reload_accounts: bool = False) -> None:
        with self._state_lock:
            self._disabled_accounts.clear()
            self._disabled_services.clear()
            if reload_accounts:
                self._accounts_cache = None
            for account in self.load_account_configs(force_reload=reload_accounts):
                self._disabled_services.setdefault(account.account_id, set())

    def get_fallback_status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "disabled_accounts": sorted(self._disabled_accounts),
                "disabled_services": {k: sorted(v) for k, v in self._disabled_services.items()},
                "accounts": [a.account_id for a in self.load_account_configs()],
                "service_order": [s.name for s in self.services],
            }

    def _available_accounts(self) -> list[AccountConfig]:
        with self._state_lock:
            return [a for a in self.load_account_configs() if a.account_id not in self._disabled_accounts]

    def _accounts_for_request(
        self,
        *,
        secret_id: str | None = None,
        secret_key: str | None = None,
        appid: str | None = None,
    ) -> list[AccountConfig]:
        if (secret_id is None) ^ (secret_key is None):
            raise TencentASRAuthError(4002, "显式传入凭证时必须至少提供 secret_id 和 secret_key")

        if secret_id and secret_key:
            return [AccountConfig("explicit", secret_id, secret_key, appid)]

        if self._default_secret_id and self._default_secret_key:
            resolved_appid = appid if appid is not None else self._default_appid
            return [AccountConfig("instance", self._default_secret_id, self._default_secret_key, resolved_appid)]
        return self._available_accounts()

    def _available_services(
        self,
        account_id: str,
        service_order: Iterable[str] | None = None,
    ) -> list[ASRService]:
        with self._state_lock:
            disabled = self._disabled_services.setdefault(account_id, set())
            order = list(service_order) if service_order else [s.name for s in self.services]
            by_name = {s.name: s for s in self.services}
            return [by_name[name] for name in order if name in by_name and name not in disabled]

    def _disable_service(self, account_id: str, service_name: str) -> None:
        with self._state_lock:
            self._disabled_services.setdefault(account_id, set()).add(service_name)
            if len(self._disabled_services[account_id]) >= len(self.services):
                self._disabled_accounts.add(account_id)

    def _disable_account(self, account_id: str) -> None:
        with self._state_lock:
            self._disabled_accounts.add(account_id)

    @staticmethod
    def _is_retryable_transient(exc: TencentASRError) -> bool:
        if isinstance(exc, (TencentASRServerError, TencentASRNetworkError, TencentASRTimeoutError)):
            return True
        return exc.code in {4006, 4008, 4009, 5000, 5001, 5002, 5003}

    @staticmethod
    def _retry_delay(attempt_index: int, base_sleep: float, max_sleep: float) -> float:
        delay = min(max_sleep, base_sleep * (2 ** max(0, attempt_index - 1)))
        return delay * (0.75 + random.random() * 0.5)

    @staticmethod
    def _call_service(service: ASRService, kind: InputKind, payload: bytes | str, account: AccountConfig) -> str:
        kwargs = {"secret_id": account.secret_id, "secret_key": account.secret_key, "appid": account.appid}
        if kind == "bytes":
            return service.by_bytes(payload, **kwargs)  # type: ignore[arg-type]
        return service.by_file(payload, **kwargs)  # type: ignore[arg-type]

    def _asr_with_fallback(
        self,
        kind: InputKind,
        payload: bytes | str,
        *,
        service_order: Iterable[str] | None = None,
        secret_id: str | None = None,
        secret_key: str | None = None,
        appid: str | None = None,
        max_retries_per_api: int = 2,
        retry_base_sleep: float = 0.35,
        retry_max_sleep: float = 2.0,
    ) -> str:
        errors: list[str] = []
        accounts = self._accounts_for_request(secret_id=secret_id, secret_key=secret_key, appid=appid)
        if not accounts:
            raise TencentASRResourceExhaustedError(f"没有可用腾讯云账户，当前状态: {self.get_fallback_status()}")

        for account in accounts:
            services = self._available_services(account.account_id, service_order=service_order)
            if not services:
                self._disable_account(account.account_id)
                continue

            for service in services:
                attempts = max(1, int(max_retries_per_api) + 1)

                for attempt in range(1, attempts + 1):
                    try:
                        return self._call_service(service, kind, payload, account)

                    except TencentASRResourceExhaustedError as exc:
                        errors.append(f"{account.account_id}/{service.name}: resource exhausted: {exc.message}")
                        self._disable_service(account.account_id, service.name)
                        break

                    except TencentASRAuthError as exc:
                        errors.append(f"{account.account_id}/{service.name}: auth/account error: {exc.message}")
                        self._disable_account(account.account_id)
                        break

                    except TencentASRError as exc:
                        errors.append(f"{account.account_id}/{service.name}: attempt {attempt}/{attempts}: {exc}")

                        if self._is_retryable_transient(exc):
                            if attempt < attempts:
                                delay = self._retry_delay(attempt, retry_base_sleep, retry_max_sleep)
                                logger.warning(
                                    "Transient Tencent ASR error on %s/%s attempt %s/%s; retry current API after %.2fs: %s",
                                    account.account_id,
                                    service.name,
                                    attempt,
                                    attempts,
                                    delay,
                                    exc,
                                )
                                time.sleep(delay)
                                continue

                            logger.warning(
                                "Transient Tencent ASR error on %s/%s exhausted %s attempts; switch to next API",
                                account.account_id,
                                service.name,
                                attempts,
                            )
                            break

                        raise

                    except Exception as exc:
                        errors.append(f"{account.account_id}/{service.name}: attempt {attempt}/{attempts}: unexpected: {exc}")
                        if attempt < attempts:
                            delay = self._retry_delay(attempt, retry_base_sleep, retry_max_sleep)
                            logger.warning(
                                "Unexpected Tencent ASR error on %s/%s attempt %s/%s; retry current API after %.2fs: %s",
                                account.account_id,
                                service.name,
                                attempt,
                                attempts,
                                delay,
                                exc,
                            )
                            time.sleep(delay)
                            continue
                        break

                if account.account_id in self._disabled_accounts:
                    break

        raise TencentASRResourceExhaustedError(
            "所有账户/API 均未成功。若错误列表里包含 transient/server/network，可稍后重试；"
            f"当前状态={self.get_fallback_status()}; errors={errors[-20:]}"
        )

    def asr_by_bytes(
        self,
        audio_data: bytes,
        *,
        service_order: Iterable[str] | None = None,
        secret_id: str | None = None,
        secret_key: str | None = None,
        appid: str | None = None,
        max_retries_per_api: int = 2,
        retry_base_sleep: float = 0.35,
        retry_max_sleep: float = 2.0,
    ) -> str:
        validate_audio_bytes(audio_data)
        return self._asr_with_fallback(
            "bytes",
            bytes(audio_data),
            service_order=service_order,
            secret_id=secret_id,
            secret_key=secret_key,
            appid=appid,
            max_retries_per_api=max_retries_per_api,
            retry_base_sleep=retry_base_sleep,
            retry_max_sleep=retry_max_sleep,
        )

    def asr_by_file(
        self,
        audio_file: str,
        *,
        service_order: Iterable[str] | None = None,
        secret_id: str | None = None,
        secret_key: str | None = None,
        appid: str | None = None,
        max_retries_per_api: int = 2,
        retry_base_sleep: float = 0.35,
        retry_max_sleep: float = 2.0,
    ) -> str:
        read_audio_file(audio_file)
        return self._asr_with_fallback(
            "file",
            audio_file,
            service_order=service_order,
            secret_id=secret_id,
            secret_key=secret_key,
            appid=appid,
            max_retries_per_api=max_retries_per_api,
            retry_base_sleep=retry_base_sleep,
            retry_max_sleep=retry_max_sleep,
        )

    def asr_by_data(
        self,
        data: Any,
        sample_rate: int = 16000,
        *,
        service_order: Iterable[str] | None = None,
        secret_id: str | None = None,
        secret_key: str | None = None,
        appid: str | None = None,
        max_retries_per_api: int = 2,
        retry_base_sleep: float = 0.35,
        retry_max_sleep: float = 2.0,
    ) -> str:
        """Transcribe numpy array / torch tensor / list as 16-bit mono WAV bytes."""
        import numpy as np

        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        data = np.asarray(data).squeeze()
        if data.ndim != 1:
            raise TencentASRError(4001, f"asr_by_data 只支持单声道一维音频，当前 shape={data.shape}")

        if np.issubdtype(data.dtype, np.floating):
            data = np.clip(data, -1.0, 1.0)
            pcm = (data * 32767.0).astype(np.int16)
        elif data.dtype == np.int16:
            pcm = data
        else:
            pcm = data.astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm.tobytes())
        return self.asr_by_bytes(
            buf.getvalue(),
            service_order=service_order,
            secret_id=secret_id,
            secret_key=secret_key,
            appid=appid,
            max_retries_per_api=max_retries_per_api,
            retry_base_sleep=retry_base_sleep,
            retry_max_sleep=retry_max_sleep,
        )

    def transcribe(
        self,
        x_cand: Any,
        sample_rate: int = 16000,
        *,
        service_order: Iterable[str] | None = None,
        secret_id: str | None = None,
        secret_key: str | None = None,
        appid: str | None = None,
        max_retries_per_api: int = 2,
        retry_base_sleep: float = 0.35,
        retry_max_sleep: float = 2.0,
    ) -> str:
        """AlibabaASR 风格统一入口。"""
        if isinstance(x_cand, bytes):
            return self.asr_by_bytes(
                x_cand,
                service_order=service_order,
                secret_id=secret_id,
                secret_key=secret_key,
                appid=appid,
                max_retries_per_api=max_retries_per_api,
                retry_base_sleep=retry_base_sleep,
                retry_max_sleep=retry_max_sleep,
            )
        if isinstance(x_cand, io.BytesIO):
            return self.asr_by_bytes(
                x_cand.getvalue(),
                service_order=service_order,
                secret_id=secret_id,
                secret_key=secret_key,
                appid=appid,
                max_retries_per_api=max_retries_per_api,
                retry_base_sleep=retry_base_sleep,
                retry_max_sleep=retry_max_sleep,
            )
        if isinstance(x_cand, str):
            if os.path.isfile(x_cand):
                return self.asr_by_file(
                    x_cand,
                    service_order=service_order,
                    secret_id=secret_id,
                    secret_key=secret_key,
                    appid=appid,
                    max_retries_per_api=max_retries_per_api,
                    retry_base_sleep=retry_base_sleep,
                    retry_max_sleep=retry_max_sleep,
                )
            raise TencentASRError(4001, f"输入字符串不是有效文件路径: {x_cand}")
        return self.asr_by_data(
            x_cand,
            sample_rate=sample_rate,
            service_order=service_order,
            secret_id=secret_id,
            secret_key=secret_key,
            appid=appid,
            max_retries_per_api=max_retries_per_api,
            retry_base_sleep=retry_base_sleep,
            retry_max_sleep=retry_max_sleep,
        )