"""Tencent ASR unified exception hierarchy and error-normalization helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass


class TencentASRError(Exception):
    """Base exception for all Tencent ASR failures.

    code follows the project-level business code convention:
      4001 invalid input/parameter/audio
      4002 auth/credential error
      4003 service not enabled
      4004 resource package exhausted
      4005 account arrears / service isolated
      4006 concurrency/rate limit
      4007 audio decode failure
      4008 timeout
      4009 connection/network failure
      5000+ server-side/transient errors
      9999 unknown/unclassified error
    """

    def __init__(
        self,
        code: int = 9999,
        message: str = "Tencent ASR error",
        *,
        service: str | None = None,
    ):
        self.code = int(code)
        self.message = str(message)
        self.service = service
        prefix = f"[{service}] " if service else ""
        super().__init__(f"{prefix}ASR Error {self.code}: {self.message}")


class TencentASRResourceExhaustedError(TencentASRError):
    def __init__(
        self,
        message: str = "资源包耗尽，请开通后付费或购买资源包",
        *,
        service: str | None = None,
    ):
        super().__init__(4004, message, service=service)


class TencentASRAuthError(TencentASRError):
    def __init__(
        self,
        code: int = 4002,
        message: str = "鉴权失败",
        *,
        service: str | None = None,
    ):
        super().__init__(code, message, service=service)


class TencentASRServerError(TencentASRError):
    def __init__(
        self,
        code: int = 5000,
        message: str = "服务端错误",
        *,
        service: str | None = None,
    ):
        super().__init__(code, message, service=service)


class TencentASRNetworkError(TencentASRError):
    def __init__(
        self,
        code: int = 4009,
        message: str = "网络异常",
        *,
        service: str | None = None,
    ):
        super().__init__(code, message, service=service)


class TencentASRTimeoutError(TencentASRNetworkError):
    def __init__(self, message: str = "请求超时", *, service: str | None = None):
        super().__init__(4008, message, service=service)


@dataclass(frozen=True)
class NormalizedASRError:
    code: int
    message: str


RESOURCE_EXHAUSTED_KEYWORDS = (
    "UserHasNoAmount",
    "UserHasNoFreeAmount",
    "resource pack exhausted",
    "amount exhausted",
    "资源包耗尽",
    "余量不足",
)

KEYWORD_CODE_MAP = {
    # client/auth/account
    "InvalidParameter": 4001,
    "InvalidParameterValue": 4001,
    "ErrorInvalidAppid": 4001,
    "ErrorInvalidVoicedata": 4001,
    "ErrorVoicedataTooLong": 4001,
    "音频数据为空": 4001,
    "参数不合法": 4001,
    "AuthFailure": 4002,
    "UnauthorizedOperation": 4002,
    "InvalidCredential": 4002,
    "鉴权失败": 4002,
    "UserNotRegistered": 4003,
    "服务未开通": 4003,
    "ServiceIsolate": 4005,
    "账户欠费": 4005,
    "LimitExceeded": 4006,
    "并发超限": 4006,
    "音频解码失败": 4007,
    "decode": 4007,
    "timeout": 4008,
    "超时": 4008,
    "连接断开": 4009,
    "Connection": 4009,
    # server/transient
    "InternalError": 5000,
    "ErrorRecognize": 5000,
    "ErrorConfigure": 5000,
    "机器负载过高": 5000,
    "网络抖动": 5001,
    "识别失败": 5002,
    "境外调用": 6001,
}


def normalize_error(code: object = None, message: object = "") -> NormalizedASRError:
    raw = f"{code or ''}: {message or ''}".strip(": ")
    lower = raw.lower()

    for kw in RESOURCE_EXHAUSTED_KEYWORDS:
        if kw.lower() in lower:
            return NormalizedASRError(4004, raw or kw)

    # Extract project-level numeric code when present.
    m = re.search(r"(?<!\d)(400[0-9]|4010|500[0-9]|6001|9999)(?!\d)", raw)
    if m:
        return NormalizedASRError(int(m.group(1)), raw)

    for kw, mapped_code in KEYWORD_CODE_MAP.items():
        if kw.lower() in lower:
            return NormalizedASRError(mapped_code, raw or kw)

    if "auth" in lower or "credential" in lower or "签名" in raw:
        return NormalizedASRError(4002, raw)
    if "network" in lower or "connection" in lower or "网络" in raw:
        return NormalizedASRError(4009, raw)
    server_error = any(
        keyword in lower for keyword in ("server", "internal")
    ) or any(keyword in raw for keyword in ("服务端", "负载"))
    if server_error:
        return NormalizedASRError(5000, raw)
    return NormalizedASRError(9999, raw or "未知 ASR 错误")


def raise_normalized_asr_error(
    code: object = None,
    message: object = "",
    *,
    service: str | None = None,
) -> None:
    err = normalize_error(code, message)
    if err.code == 4004:
        raise TencentASRResourceExhaustedError(err.message, service=service)
    if err.code in (4002, 4003, 4005):
        raise TencentASRAuthError(err.code, err.message, service=service)
    if err.code == 4008:
        raise TencentASRTimeoutError(err.message, service=service)
    if err.code == 4009:
        raise TencentASRNetworkError(4009, err.message, service=service)
    if 5000 <= err.code <= 5999:
        raise TencentASRServerError(err.code, err.message, service=service)
    raise TencentASRError(err.code, err.message, service=service)


def validate_audio_bytes(audio_data: bytes, *, max_size_mb: int | None = None) -> None:
    if not isinstance(audio_data, (bytes, bytearray)) or len(audio_data) == 0:
        raise TencentASRError(4001, "音频数据为空或不是 bytes")
    if max_size_mb is not None and len(audio_data) > max_size_mb * 1024 * 1024:
        raise TencentASRError(4001, f"音频数据超过 {max_size_mb}MB 限制")
