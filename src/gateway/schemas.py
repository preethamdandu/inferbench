from pydantic import BaseModel, Field


class GatewayGenerateParams(BaseModel):
    model: str = "default"
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str | None = None


class Timing(BaseModel):
    total_ms: float
    time_to_first_token_ms: float | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: list[Choice]
    usage: Usage
    timing: Timing


class Delta(BaseModel):
    text: str


class StreamChoice(BaseModel):
    delta: Delta
    index: int = 0
    finish_reason: str | None = None


class StreamResponse(BaseModel):
    id: str
    object: str = "text_completion_chunk"
    model: str
    choices: list[StreamChoice]


class GatewayImageParams(BaseModel):
    prompt: str
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    seed: int = 42


class ImageResponse(BaseModel):
    image_b64: str
    total_time_ms: float
    avg_step_time_ms: float
    gpu_peak_memory_mb: float


# Aliases for canonical import names
CompletionRequest = GatewayGenerateParams
