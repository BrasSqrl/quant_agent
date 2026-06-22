from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class AppClientError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class AgentAppClient(Protocol):
    def create_preflight(
        self,
        *,
        app_id: str,
        capability_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class LocalAgentAppClient:
    quant_data_base_url: str
    timeout_seconds: float = 5.0

    @classmethod
    def from_environment(cls) -> "LocalAgentAppClient":
        return cls(
            quant_data_base_url=os.environ.get(
                "QUANT_DATA_AGENT_API_BASE_URL",
                "http://127.0.0.1:8830",
            ).rstrip("/")
        )

    def create_preflight(
        self,
        *,
        app_id: str,
        capability_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if app_id != "quant_data":
            raise AppClientError(f"No local preflight client is configured for {app_id}.", status_code=422)
        url = f"{self.quant_data_base_url}/api/agent/actions/{capability_id}/preflight"
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AppClientError(
                f"Preflight app returned HTTP {exc.code}: {detail[:240]}",
                status_code=502,
            ) from exc
        except (OSError, URLError) as exc:
            raise AppClientError(
                "Quant Data preflight app is unavailable.",
                status_code=503,
            ) from exc

        try:
            payload_object = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise AppClientError("Preflight app returned invalid JSON.", status_code=502) from exc
        if not isinstance(payload_object, dict):
            raise AppClientError("Preflight app returned a non-object response.", status_code=502)
        return payload_object
