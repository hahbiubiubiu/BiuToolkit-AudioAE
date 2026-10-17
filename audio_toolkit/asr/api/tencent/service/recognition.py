"""Tencent async task ASR API wrapper based on tencentcloud-sdk-python."""
from __future__ import annotations

import base64
import json
import os
import time

from tencentcloud.asr.v20190614 import asr_client, models
from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile

from ..exception import raise_normalized_asr_error, validate_audio_bytes

SERVICE_NAME = "task"
ENGINE_MODEL_TYPE = "16k_en"
TIMEOUT = 30
MAX_POLL = 60
POLL_INTERVAL = 0.1


def _client(
    *,
    secret_id: str | None = None,
    secret_key: str | None = None,
    appid: str | None = None,
) -> asr_client.AsrClient:
    del appid
    secret_id = secret_id or os.getenv("TENCENTCLOUD_SECRET_ID")
    secret_key = secret_key or os.getenv("TENCENTCLOUD_SECRET_KEY")
    if not secret_id or not secret_key:
        raise_normalized_asr_error(
            4002,
            "缺少 TENCENTCLOUD_SECRET_ID/TENCENTCLOUD_SECRET_KEY",
            service=SERVICE_NAME,
        )
    try:
        cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile(
            endpoint="asr.tencentcloudapi.com",
            reqTimeout=TIMEOUT,
        )
        client_profile = ClientProfile(httpProfile=http_profile)
        return asr_client.AsrClient(cred, "", client_profile)
    except TencentCloudSDKException as e:
        raise_normalized_asr_error(e.code, e.message, service=SERVICE_NAME)


def _to_base64(audio_data: bytes) -> tuple[str, int]:
    validate_audio_bytes(audio_data, max_size_mb=5)
    return base64.b64encode(audio_data).decode("utf-8"), len(audio_data)


def _create_task(client: asr_client.AsrClient, audio_data: bytes) -> int:
    data, data_len = _to_base64(audio_data)
    req = models.CreateRecTaskRequest()
    req.from_json_string(
        json.dumps(
            {
                "EngineModelType": ENGINE_MODEL_TYPE,
                "ChannelNum": 1,
                "ResTextFormat": 0,
                "SourceType": 1,
                "Data": data,
                "DataLen": data_len,
            }
        )
    )
    try:
        resp = client.CreateRecTask(req)
        return resp.Data.TaskId
    except TencentCloudSDKException as e:
        raise_normalized_asr_error(e.code, e.message, service=SERVICE_NAME)


def _parse_result(result: str | None) -> str:
    if not result:
        return ""
    return result.split("]")[-1].strip()


def _poll(client: asr_client.AsrClient, task_id: int) -> str:
    for _ in range(MAX_POLL):
        req = models.DescribeTaskStatusRequest()
        req.from_json_string(json.dumps({"TaskId": task_id}))
        try:
            resp = client.DescribeTaskStatus(req)
        except TencentCloudSDKException as e:
            raise_normalized_asr_error(e.code, e.message, service=SERVICE_NAME)

        data = resp.Data
        if data.Status == 2:
            return _parse_result(data.Result)
        if data.Status == 3:
            raise_normalized_asr_error(
                None,
                data.ErrorMsg or "任务识别失败",
                service=SERVICE_NAME,
            )
        time.sleep(POLL_INTERVAL)
    raise_normalized_asr_error(4008, "异步任务轮询超时", service=SERVICE_NAME)


def asr_by_bytes(
    audio_data: bytes,
    *,
    secret_id: str | None = None,
    secret_key: str | None = None,
    appid: str | None = None,
) -> str:
    validate_audio_bytes(audio_data, max_size_mb=5)
    client = _client(secret_id=secret_id, secret_key=secret_key, appid=appid)
    task_id = _create_task(client, audio_data)
    return _poll(client, task_id)
