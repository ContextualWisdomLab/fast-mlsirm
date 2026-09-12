import json
from dataclasses import replace

import pytest

from test_rubric_generation import _request
import fast_mlsirm.rubric.generation as generation


def test_public_generation_request_rejects_float_overflow() -> None:
    request = _request()
    assert request.to_provider_dict()["contract_id"] == request.contract_id

    contract_values = json.loads(request.contract_json)
    contract_values["extra"] = float("inf")
    contract_fingerprint = generation._sha256_hex(
        generation._contract_body(contract_values)
    )
    contract_values.update(
        contract_fingerprint=contract_fingerprint,
        contract_id=f"generation_contract_{contract_fingerprint[:16]}",
        contract_handle=f"generation_contract_{contract_fingerprint[:32]}",
    )
    contract_json = json.dumps(
        contract_values, separators=(",", ":"), allow_nan=True
    ).replace(
        '"extra":Infinity', '"extra":1e999'
    )
    assert '"extra":1e999' in contract_json
    identity = generation._request_identity_payload(
        contract_values,
        request.blueprint,
        request.sources,
        request.generation_seed,
        request.schema_version,
    )
    request_identity = generation._sha256_hex(identity)
    request_id = f"generation_request_{request_identity[:16]}"
    with pytest.raises(ValueError, match="valid JSON text"):
        replace(
            request,
            request_id=request_id,
            contract_id=contract_values["contract_id"],
            contract_json=contract_json,
        )
