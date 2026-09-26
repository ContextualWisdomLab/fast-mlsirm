import json
from dataclasses import replace

import pytest

from test_rubric_generation import _request
import fast_mlsirm.rubric.generation as generation_contracts


def test_public_generation_request_rejects_float_overflow() -> None:
    generation_request = _request()
    assert generation_request.to_provider_dict()["contract_id"] == generation_request.contract_id

    contract_values = json.loads(generation_request.contract_json)
    contract_values["extra"] = float("inf")
    contract_fingerprint = generation_contracts._sha256_hex(
        generation_contracts._contract_body(contract_values)
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
    request_identity_payload = generation_contracts._request_identity_payload(
        contract_values,
        generation_request.blueprint,
        generation_request.sources,
        generation_request.generation_seed,
        generation_request.schema_version,
    )
    request_identity = generation_contracts._sha256_hex(request_identity_payload)
    request_id = f"generation_request_{request_identity[:16]}"
    with pytest.raises(ValueError, match="valid JSON text"):
        replace(
            generation_request,
            request_id=request_id,
            contract_id=contract_values["contract_id"],
            contract_json=contract_json,
        )
