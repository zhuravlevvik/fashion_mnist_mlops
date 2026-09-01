import json
import os
from pathlib import Path

import httpx
import pytest

SCENARIO_PATH = Path(__file__).resolve().parents[2] / "scenario.json"


def expand_payload(payload: dict | None) -> dict | None:
    if not payload:
        return payload
    expanded = dict(payload)
    pixels = expanded.get("pixels")
    if isinstance(pixels, dict) and {"fill", "size"}.issubset(pixels):
        expanded["pixels"] = [pixels["fill"]] * pixels["size"]
    return expanded


@pytest.mark.functional
def test_api_scenarios() -> None:
    specifications = json.loads(SCENARIO_PATH.read_text())
    base_url = os.getenv("API_BASE_URL", specifications["base_url"])

    with httpx.Client(base_url=base_url, timeout=15) as client:
        for scenario in specifications["scenarios"]:
            response = client.request(
                scenario["method"],
                scenario["path"],
                json=expand_payload(scenario.get("payload"))
            )

            assert response.status_code == scenario["expected_status"]
            body = response.json()
            assert set(scenario["expected_fields"]).issubset(body)
