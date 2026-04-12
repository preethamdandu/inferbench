import pytest
from pydantic import ValidationError
from src.gateway.schemas import GatewayGenerateParams


def test_gateway_generate_params_valid():
    params = GatewayGenerateParams(prompt="Hello", max_tokens=100, temperature=0.5)
    assert params.max_tokens == 100


def test_gateway_generate_params_invalid():
    with pytest.raises(ValidationError):
        GatewayGenerateParams(prompt="Hello", max_tokens=5000)  # Assuming max_tokens limit is 4096
