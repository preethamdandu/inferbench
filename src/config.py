from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for InferBench using environment variables via pydantic-settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gateway Config
    gateway_port: int = 8000
    gateway_host: str = "0.0.0.0"

    # Backend URLs
    vllm_url: str = "http://localhost:8001"
    tgi_url: str = "http://localhost:8002"
    custom_url: str = "http://localhost:8003"

    # Model configs
    llm_model_id: str = "google/gemma-2-2b-it"
    diffusion_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # Custom Backend Config
    max_batch_size: int = 32
    kv_cache_blocks: int = 128
    block_size: int = 16

    # Benchmarking
    benchmark_duration_seconds: int = 60
    warmup_requests: int = 5
    concurrency_levels: list[int] = [1, 2, 4, 8, 16, 32]


settings = Settings()
