from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Config dùng chung cho tất cả sub-settings
# Mỗi sub-class là BaseSettings độc lập → cần env_file để tự đọc .env
_ENV_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class AppSettings(BaseSettings):
    model_config = _ENV_CONFIG

    name: str = Field(default="ERP Agentic Chatbot", alias="APP_NAME")
    version: str = Field(default="0.1.0", alias="APP_VERSION")
    env: Literal["development", "staging", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    port: int = Field(default=8000, alias="APP_PORT")
    secret_key: str = Field(default="dev-secret-key", alias="APP_SECRET_KEY")
    debug: bool = Field(default=False, alias="DEBUG")
    cors_origins: list[str] = Field(default=["*"], alias="APP_CORS_ORIGINS")

    @property
    def is_production(self) -> bool:
        return self.env == "production"


class LLMSettings(BaseSettings):
    model_config = _ENV_CONFIG

    provider: Literal["anthropic", "openai", "gemini", "ollama"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    ollama_api_key: str = Field(default="", alias="OLLAMA_API_KEY")

    # base_url: để trống = dùng endpoint mặc định của provider.
    # Set khi gọi model qua proxy / provider khác (vd: Together, Groq, OpenRouter)
    base_url: str = Field(default="", alias="LLM_BASE_URL")

    model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    max_tool_iterations: int = Field(default=10, alias="LLM_MAX_TOOL_ITERATIONS")

    @field_validator("max_tool_iterations")
    @classmethod
    def validate_iterations(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError("max_tool_iterations phải từ 1 đến 50")
        return v

    def get_api_key(self) -> str:
        keys = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
            "ollama": self.ollama_api_key,
        }
        key = keys[self.provider]
        # Ollama doesn't require an API key, so we can just return an empty string.
        if self.provider == "ollama":
            return key
        if not key:
            raise ValueError(f"Thiếu API key cho provider '{self.provider}'")
        return key


class ERPDatabaseSettings(BaseSettings):
    model_config = _ENV_CONFIG

    host: str = Field(default="localhost", alias="ERP_DB_HOST")
    port: int = Field(default=5432, alias="ERP_DB_PORT")
    name: str = Field(default="erp_chatbot", alias="ERP_DB_NAME")
    user: str = Field(default="erp_user", alias="ERP_DB_USER")
    password: str = Field(default="erp_secret", alias="ERP_DB_PASSWORD")
    pool_size: int = Field(default=10, alias="ERP_DB_POOL_SIZE")
    pool_timeout: int = Field(default=30, alias="ERP_DB_POOL_TIMEOUT")

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def url_sync(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class VectorStoreSettings(BaseSettings):
    model_config = _ENV_CONFIG

    provider: Literal["chroma", "qdrant", "pgvector"] = Field(
        default="qdrant", alias="VECTOR_STORE_PROVIDER"
    )
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8001, alias="CHROMA_PORT")
    chroma_collection: str = Field(default="erp_docs", alias="CHROMA_COLLECTION")

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="erp_docs", alias="QDRANT_COLLECTION")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")

    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")


class RedisSettings(BaseSettings):
    model_config = _ENV_CONFIG

    host: str = Field(default="localhost", alias="REDIS_HOST")
    port: int = Field(default=6379, alias="REDIS_PORT")
    password: str = Field(default="", alias="REDIS_PASSWORD")
    db: int = Field(default=0, alias="REDIS_DB")
    session_ttl: int = Field(default=86400, alias="REDIS_SESSION_TTL")
    cache_ttl: int = Field(default=300, alias="REDIS_CACHE_TTL")

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class AuthSettings(BaseSettings):
    model_config = _ENV_CONFIG

    jwt_secret_key: str = Field(default="dev-jwt-secret", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=60, alias="JWT_EXPIRE_MINUTES")
    api_key_header: str = Field(default="X-API-Key", alias="API_KEY_HEADER")


class RateLimitSettings(BaseSettings):
    model_config = _ENV_CONFIG

    per_minute: int = Field(default=30, alias="RATE_LIMIT_PER_MINUTE")
    per_hour: int = Field(default=500, alias="RATE_LIMIT_PER_HOUR")


class LoggingSettings(BaseSettings):
    model_config = _ENV_CONFIG

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    format: Literal["json", "text"] = Field(default="json", alias="LOG_FORMAT")
    file: str = Field(default="", alias="LOG_FILE")


class MCPSettings(BaseSettings):
    model_config = _ENV_CONFIG

    host: str = Field(default="localhost", alias="MCP_SERVER_HOST")
    port: int = Field(default=8100, alias="MCP_SERVER_PORT")
    timeout: int = Field(default=30, alias="MCP_TIMEOUT")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class Settings(BaseSettings):
    """Root settings — import cái này ở khắp nơi."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    erp_db: ERPDatabaseSettings = Field(default_factory=ERPDatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)


@lru_cache
def get_settings() -> Settings:
    """
    Trả về Settings singleton — cache lại sau lần đầu load.

    Dùng:
        from config.settings import get_settings
        settings = get_settings()
        print(settings.llm.model)
    """
    return Settings()
