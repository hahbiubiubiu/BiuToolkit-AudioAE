from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from audio_toolkit.asr.api.base import ASRAPI

from .exception import (
    TencentASRError,
    TencentASRAuthError,
    TencentASRNetworkError,
    TencentASRResourceExhaustedError,
    TencentASRServerError,
    validate_audio_bytes,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccountConfig:
    account_id: str
    secret_id: str
    secret_key: str
    appid: str | None = None


@dataclass(frozen=True)
class ASRService:
    name: str
    recognize: Callable[..., str]


def _default_services() -> tuple[ASRService, ...]:
    from .service import (
        recognition,
        recognition_flash,
        recognition_realtime,
        recognition_sentence,
    )

    return (
        ASRService("flash", recognition_flash.asr_by_bytes),
        ASRService("sentence", recognition_sentence.asr_by_bytes),
        ASRService("realtime", recognition_realtime.asr_by_bytes),
        ASRService("task", recognition.asr_by_bytes),
    )


class TencentASR(ASRAPI):
    """Tencent ASR client with automatic service and account switching."""

    def __init__(
        self,
        services: Iterable[ASRService] | None = None,
        accounts: Iterable[AccountConfig] | None = None,
        *,
        secret_id: str | None = None,
        secret_key: str | None = None,
        appid: str | None = None,
        max_retries: int = 2,
        retry_base_sleep: float = 0.35,
        retry_max_sleep: float = 2.0,
        env_file: str | os.PathLike[str] | None = None,
        load_env: bool = True,
    ):
        if (secret_id is None) != (secret_key is None):
            raise TencentASRAuthError(
                message="secret_id and secret_key must be provided together"
            )
        if accounts is not None and secret_id is not None:
            raise ValueError("Use accounts or instance credentials, not both")

        self.services = tuple(services) if services is not None else _default_services()
        if not self.services:
            raise ValueError("At least one Tencent ASR service is required")

        if accounts is not None:
            account_cache = list(accounts)
        elif secret_id is not None and secret_key is not None:
            account_cache = [
                AccountConfig("instance", secret_id, secret_key, appid)
            ]
        else:
            account_cache = None

        super().__init__(
            env_file=env_file,
            load_env=load_env and account_cache is None,
        )

        self.max_retries = max(0, int(max_retries))
        self.retry_base_sleep = retry_base_sleep
        self.retry_max_sleep = retry_max_sleep
        self._accounts_cache = account_cache
        self._disabled_accounts: set[str] = set()
        self._disabled_services: dict[str, set[str]] = {}
        self._state_lock = threading.RLock()
        self._initialize_service_state(account_cache or [])

    def load_account_configs(
        self,
        *,
        force_reload: bool = False,
    ) -> list[AccountConfig]:
        with self._state_lock:
            if self._accounts_cache is not None and (
                not force_reload or not self._load_env
            ):
                return list(self._accounts_cache)

            if self._load_env and force_reload:
                self._load_environment()

            accounts = self._accounts_from_env()
            if not accounts:
                raise TencentASRAuthError(
                    message="Tencent ASR account credentials are not configured"
                )

            self._accounts_cache = accounts
            self._initialize_service_state(accounts)
            return list(accounts)

    @staticmethod
    def _accounts_from_env() -> list[AccountConfig]:
        accounts = []
        index = 1
        while True:
            secret_id = os.getenv(f"ACCOUNT_SECRET_ID_{index}")
            secret_key = os.getenv(f"ACCOUNT_SECRET_KEY_{index}")
            appid = os.getenv(f"ACCOUNT_APP_ID_{index}")
            if not any((secret_id, secret_key, appid)):
                break
            if secret_id and secret_key:
                accounts.append(
                    AccountConfig(
                        f"account_{index}",
                        secret_id.strip(),
                        secret_key.strip(),
                        appid.strip() if appid else None,
                    )
                )
            else:
                logger.warning("Ignoring incomplete Tencent account_%s", index)
            index += 1

        if accounts:
            return accounts

        secret_id = os.getenv("TENCENTCLOUD_SECRET_ID")
        secret_key = os.getenv("TENCENTCLOUD_SECRET_KEY")
        appid = os.getenv("APPID")
        if secret_id and secret_key:
            return [
                AccountConfig(
                    "default",
                    secret_id.strip(),
                    secret_key.strip(),
                    appid.strip() if appid else None,
                )
            ]
        return []

    def reset_switch_state(self, *, reload_accounts: bool = False) -> None:
        with self._state_lock:
            self._disabled_accounts.clear()
            self._disabled_services.clear()
            if reload_accounts and self._load_env:
                self._accounts_cache = None
            self._initialize_service_state(
                self.load_account_configs(force_reload=reload_accounts)
            )

    def get_switch_status(self) -> dict[str, object]:
        with self._state_lock:
            accounts = self.load_account_configs()
            return {
                "accounts": [account.account_id for account in accounts],
                "services": [service.name for service in self.services],
                "disabled_accounts": sorted(self._disabled_accounts),
                "disabled_services": {
                    account.account_id: sorted(
                        self._disabled_services.get(account.account_id, set())
                    )
                    for account in accounts
                },
            }

    def _initialize_service_state(
        self,
        accounts: Iterable[AccountConfig],
    ) -> None:
        for account in accounts:
            self._disabled_services.setdefault(account.account_id, set())

    def _active_accounts(self) -> list[AccountConfig]:
        with self._state_lock:
            return [
                account
                for account in self.load_account_configs()
                if account.account_id not in self._disabled_accounts
            ]

    def _active_services(self, account_id: str) -> list[ASRService]:
        with self._state_lock:
            disabled = self._disabled_services.setdefault(account_id, set())
            return [
                service
                for service in self.services
                if service.name not in disabled
            ]

    def _disable_service(self, account_id: str, service_name: str) -> None:
        with self._state_lock:
            disabled = self._disabled_services.setdefault(account_id, set())
            disabled.add(service_name)
            if len(disabled) == len(self.services):
                self._disabled_accounts.add(account_id)

    def _disable_account(self, account_id: str) -> None:
        with self._state_lock:
            self._disabled_accounts.add(account_id)

    def _transcribe_bytes(
        self,
        audio_data: bytes,
        sample_rate: int | None,
    ) -> str:
        del sample_rate
        validate_audio_bytes(audio_data)
        failures = []

        for account in self._active_accounts():
            for service in self._active_services(account.account_id):
                try:
                    return self._call_with_retries(service, account, audio_data)
                except TencentASRResourceExhaustedError as exc:
                    failures.append(self._failure(account, service, exc))
                    self._disable_service(account.account_id, service.name)
                except TencentASRAuthError as exc:
                    failures.append(self._failure(account, service, exc))
                    self._disable_account(account.account_id)
                    break
                except TencentASRError as exc:
                    if not self._is_transient(exc):
                        raise
                    failures.append(self._failure(account, service, exc))

        raise TencentASRResourceExhaustedError(
            "All Tencent ASR accounts and services failed: "
            + "; ".join(failures)
        )

    def _call_with_retries(
        self,
        service: ASRService,
        account: AccountConfig,
        audio_data: bytes,
    ) -> str:
        attempt = 0
        while True:
            try:
                return service.recognize(
                    audio_data,
                    secret_id=account.secret_id,
                    secret_key=account.secret_key,
                    appid=account.appid,
                )
            except TencentASRError as exc:
                if not self._is_transient(exc) or attempt >= self.max_retries:
                    raise
                delay = min(
                    self.retry_max_sleep,
                    self.retry_base_sleep * 2**attempt,
                )
                if delay > 0:
                    time.sleep(delay)
                attempt += 1

    @staticmethod
    def _is_transient(error: TencentASRError) -> bool:
        return isinstance(
            error,
            (TencentASRNetworkError, TencentASRServerError),
        ) or error.code in {4006, 4008, 4009, 5000, 5001, 5002, 5003}

    @staticmethod
    def _failure(
        account: AccountConfig,
        service: ASRService,
        error: TencentASRError,
    ) -> str:
        return f"{account.account_id}/{service.name}: {error.message}"
