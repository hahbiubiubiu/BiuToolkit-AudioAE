"""Tencent SentenceRecognition API wrapper."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

import requests

from ..exception import (
    TencentASRError,
    TencentASRNetworkError,
    TencentASRServerError,
    TencentASRTimeoutError,
    raise_normalized_asr_error,
    validate_audio_bytes,
)

SERVICE_NAME = "sentence"
SERVICE = "asr"
HOST = "asr.tencentcloudapi.com"
ENDPOINT = f"https://{HOST}"
REGION = "ap-guangzhou"
ACTION = "SentenceRecognition"
VERSION = "2019-06-14"
ALGORITHM = "TC3-HMAC-SHA256"
TIMEOUT = 30


def _credential(
    *,
    secret_id: str | None = None,
    secret_key: str | None = None,
    appid: str | None = None,
) -> tuple[str, str]:
    del appid
    secret_id = secret_id or os.getenv("TENCENTCLOUD_SECRET_ID")
    secret_key = secret_key or os.getenv("TENCENTCLOUD_SECRET_KEY")
    if not secret_id or not secret_key:
        raise_normalized_asr_error(
            4002,
            "缺少 TENCENTCLOUD_SECRET_ID/TENCENTCLOUD_SECRET_KEY",
            service=SERVICE_NAME,
        )
    return secret_id, secret_key


def _sign(secret_key: str, payload: str, timestamp: int) -> tuple[str, str, str]:
    date = time.strftime("%Y-%m-%d", time.gmtime(timestamp))
    ct = "application/json; charset=utf-8"
    canonical_headers = (
        f"content-type:{ct}\nhost:{HOST}\n"
        f"x-tc-action:{ACTION.lower()}\n"
    )
    signed_headers = "content-type;host;x-tc-action"
    hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = "\n".join(
        ["POST", "/", "", canonical_headers, signed_headers, hashed_payload]
    )
    credential_scope = f"{date}/{SERVICE}/tc3_request"
    hashed_canonical_request = hashlib.sha256(
        canonical_request.encode("utf-8")
    ).hexdigest()
    string_to_sign = (
        f"{ALGORITHM}\n{timestamp}\n{credential_scope}\n"
        f"{hashed_canonical_request}"
    )

    def sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = sign(secret_date, SERVICE)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(
        secret_signing,
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return credential_scope, signed_headers, signature


def asr_by_bytes(
    audio_data: bytes,
    *,
    secret_id: str | None = None,
    secret_key: str | None = None,
    appid: str | None = None,
) -> str:
    validate_audio_bytes(audio_data, max_size_mb=3)
    secret_id, secret_key = _credential(
        secret_id=secret_id,
        secret_key=secret_key,
        appid=appid,
    )

    params = {
        "ProjectId": 0,
        "SubServiceType": 2,
        "EngSerViceType": "16k_en",
        "SourceType": 1,
        "VoiceFormat": "wav",
        "Data": base64.b64encode(audio_data).decode("utf-8"),
        "DataLen": len(audio_data),
    }
    payload = json.dumps(params, separators=(",", ":"))
    timestamp = int(time.time())
    credential_scope, signed_headers, signature = _sign(secret_key, payload, timestamp)
    authorization = (
        f"{ALGORITHM} Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json; charset=utf-8",
        "Host": HOST,
        "X-TC-Action": ACTION,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": VERSION,
        "X-TC-Region": REGION,
    }

    try:
        resp = requests.post(ENDPOINT, data=payload, headers=headers, timeout=TIMEOUT)
    except requests.exceptions.Timeout as e:
        raise TencentASRTimeoutError(
            "一句话识别请求超时",
            service=SERVICE_NAME,
        ) from e
    except requests.exceptions.ConnectionError as e:
        raise TencentASRNetworkError(
            4009,
            "一句话识别连接失败",
            service=SERVICE_NAME,
        ) from e
    except requests.RequestException as e:
        raise TencentASRNetworkError(
            4009,
            f"一句话识别网络异常: {e}",
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
            f"一句话识别响应不是 JSON: {resp.text[:300]}",
            service=SERVICE_NAME,
        ) from e

    err = data.get("Response", {}).get("Error")
    if err:
        raise_normalized_asr_error(
            err.get("Code"),
            err.get("Message", err),
            service=SERVICE_NAME,
        )
    return data.get("Response", {}).get("Result", "") or ""
