import logging
import sys
from typing import Any

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


def setup_logging(level: str = "INFO", fmt: str = "json", log_file: str = "") -> None:
    """
    Khởi tạo logging toàn app. Gọi 1 lần duy nhất trong main.py.

    Luôn dùng dual output:
      - Console (stdout): ngắn gọn, có màu, dễ nhìn nhanh
      - File (nếu có):    JSON chi tiết, dễ search/grep/parse
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if HAS_STRUCTLOG:
        _setup_dual_logging(log_level, log_file)
    else:
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        if log_file:
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        _setup_stdlib(log_level, handlers)


def _setup_dual_logging(level: int, log_file: str) -> None:
    """
    Console: ngắn gọn, có màu (ConsoleRenderer)
    File:    JSON chi tiết (JSONRenderer)

    Dùng structlog foreign_pre_chain + stdlib ProcessorFormatter
    để mỗi handler có renderer riêng.
    """
    import structlog

    # Shared processors — chạy trước khi tới formatter
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
        # Chuyển structlog event → stdlib LogRecord
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # ── Console formatter: ngắn gọn, có màu ──
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(colors=True, pad_event=40),
        ],
    )

    # ── File formatter: JSON chi tiết ──
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )

    # ── Handlers ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    handlers: list[logging.Handler] = [console_handler]

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # ── Root logger ──
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)

    # Giảm noise từ thư viện bên ngoài
    for noisy in ("httpx", "httpcore", "uvicorn.access", "openai._base_client", "fastembed"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # ── Structlog config ──
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def _setup_stdlib(level: int, handlers: list[logging.Handler]) -> None:
    text_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, handlers=handlers, format=text_fmt)

    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> Any:
    """
    Trả về logger phù hợp với config hiện tại.

    Dùng:
        from config.logging import get_logger
        logger = get_logger(__name__)
        logger.info("tool called", tool="erp_query", user_id="u123")
    """
    if HAS_STRUCTLOG:
        import structlog
        return structlog.get_logger(name)
    return logging.getLogger(name)
