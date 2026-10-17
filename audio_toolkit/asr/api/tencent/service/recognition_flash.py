"""Tencent Flash ASR API wrapper.

Low-level API contract:
    asr_by_bytes(audio_data, *, secret_id=None, secret_key=None, appid=None) -> str
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from urllib.parse import urlencode

import requests

from ..exception import (
    TencentASRError,
    TencentASRNetworkError,
    TencentASRServerError,
    TencentASRTimeoutError,
    raise_normalized_asr_error,
    validate_audio_bytes,
)

SERVICE_NAME = "flash"
ENGINE_TYPE = "16k_en"
VOICE_FORMAT = "wav"
TIMEOUT = 30


def _credential(
    *,
    secret_id: str | None = None,
    secret_key: str | None = None,
    appid: str | None = None,
) -> tuple[str, str, str]:
    appid = appid or os.getenv("APPID")
    secret_id = secret_id or os.getenv("TENCENTCLOUD_SECRET_ID")
    secret_key = secret_key or os.getenv("TENCENTCLOUD_SECRET_KEY")
    if not all([appid, secret_id, secret_key]):
        raise_normalized_asr_error(
            4002,
            "缺少 APPID/TENCENTCLOUD_SECRET_ID/TENCENTCLOUD_SECRET_KEY",
            service=SERVICE_NAME,
        )
    return appid, secret_id, secret_key


def asr_by_bytes(
    audio_data: bytes,
    *,
    secret_id: str | None = None,
    secret_key: str | None = None,
    appid: str | None = None,
) -> str:
    validate_audio_bytes(audio_data)
    appid, secret_id, secret_key = _credential(
        secret_id=secret_id,
        secret_key=secret_key,
        appid=appid,
    )

    endpoint = f"asr.cloud.tencent.com/asr/flash/v1/{appid}"
    params = {
        "engine_type": ENGINE_TYPE,
        "voice_format": VOICE_FORMAT,
        "secretid": secret_id,
        "timestamp": int(time.time()),
    }
    query_str = urlencode(sorted(params.items()))
    signing_context = f"POST{endpoint}?{query_str}"
    signature = base64.b64encode(
        hmac.new(
            secret_key.encode("utf-8"),
            signing_context.encode("utf-8"),
            hashlib.sha1,
        ).digest()
    ).decode("utf-8")

    try:
        resp = requests.post(
            f"https://{endpoint}?{query_str}",
            headers={
                "Authorization": signature,
                "Content-Type": "application/octet-stream",
            },
            data=audio_data,
            timeout=TIMEOUT,
        )
    except requests.exceptions.Timeout as e:
        raise TencentASRTimeoutError(
            "极速识别请求超时",
            service=SERVICE_NAME,
        ) from e
    except requests.exceptions.ConnectionError as e:
        raise TencentASRNetworkError(
            4009,
            "极速识别连接失败",
            service=SERVICE_NAME,
        ) from e
    except requests.RequestException as e:
        raise TencentASRNetworkError(
            4009,
            f"极速识别网络异常: {e}",
            service=SERVICE_NAME,
        ) from e

    if resp.status_code >= 500:
        raise TencentASRServerError(
            5000,
            f"HTTP {resp.status_code}: {resp.text}",
            service=SERVICE_NAME,
        )
    if resp.status_code != 200:
        raise_normalized_asr_error(resp.status_code, resp.text, service=SERVICE_NAME)

    try:
        data = resp.json()
    except ValueError as e:
        raise TencentASRError(
            9999,
            f"极速识别响应不是 JSON: {resp.text[:300]}",
            service=SERVICE_NAME,
        ) from e

    code = data.get("code", 0)
    if code != 0:
        raise_normalized_asr_error(
            code,
            data.get("message", data),
            service=SERVICE_NAME,
        )

    flash_result = data.get("flash_result") or []
    return " ".join(seg.get("text", "") for seg in flash_result).strip()
