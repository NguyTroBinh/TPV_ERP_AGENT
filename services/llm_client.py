from typing import Any, AsyncIterator, Union
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    SystemMessage,
    ToolMessage,
)

from config import get_settings, get_logger
from config.constants import Role, StopReason
from services.ollama_service import OllamaService

logger = get_logger(__name__)
settings = get_settings()


def _to_lc_message(msg: dict) -> BaseMessage:
    """Chuyển message dict (session history) → LangChain BaseMessage."""
    role = msg.get("role")
    content = msg.get("content", "")

    if role == Role.USER:
        return HumanMessage(content=content)
    if role == Role.ASSISTANT:
        # Giữ lại tool_calls nếu có (cần cho ReAct loop)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            return AIMessage(content=content, tool_calls=tool_calls)
        return AIMessage(content=content)
    if role == Role.SYSTEM:
        return SystemMessage(content=content)
    if role == Role.TOOL:
        return ToolMessage(
            content=content,
            tool_call_id=msg.get("tool_call_id", ""),
        )
    # fallback
    return HumanMessage(content=content)


def _to_ollama_message(msg: dict) -> dict:
    """Chuyển message dict → Ollama-compatible dict {role, content}."""
    role = msg.get("role", "user")
    # Ollama không có role "tool" — map về "user" với prefix
    if role == Role.TOOL:
        return {"role": "user", "content": f"[Tool result] {msg.get('content', '')}"}
    return {"role": role, "content": msg.get("content", "")}


def _normalize_ollama_tool_calls(raw_tool_calls) -> list[dict]:
    """
    Normalize tool_calls từ Ollama response → list[dict] chuẩn cho AIMessage.
    Ollama trả về list ToolCall objects (pydantic), LangChain cần list dict.
    """
    if not raw_tool_calls:
        return []
    result = []
    for tc in raw_tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            result.append({
                "id": tc.get("id", ""),
                "name": fn.get("name", "") if isinstance(fn, dict) else "",
                "args": fn.get("arguments", {}) if isinstance(fn, dict) else {},
            })
        else:
            # ToolCall pydantic object: .function.name / .function.arguments
            fn = getattr(tc, "function", None)
            result.append({
                "id": getattr(tc, "id", "") or "",
                "name": getattr(fn, "name", "") if fn else "",
                "args": getattr(fn, "arguments", {}) if fn else {},
            })
    return result


def _parse_stop_reason(response: AIMessage) -> StopReason:
    """Đọc finish_reason từ response metadata."""
    reason = (
        response.response_metadata.get("finish_reason")
        or response.response_metadata.get("stop_reason")
        or ""
    )
    if reason == "tool_calls":
        return StopReason.TOOL_USE
    if reason == "length":
        return StopReason.MAX_TOKENS
    return StopReason.END_TURN


class LLMClient:
    """
    Wrapper mỏng quanh LangChain ChatOpenAI.

    Hỗ trợ mọi provider có OpenAI-compatible API:
      - OpenAI:      LLM_PROVIDER=openai,    LLM_BASE_URL=(để trống)
      - Anthropic*:  LLM_PROVIDER=anthropic, LLM_BASE_URL=(để trống)  [qua langchain-openai compat]
      - OpenRouter:  LLM_PROVIDER=openai,    LLM_BASE_URL=https://openrouter.ai/api/v1
      - Together:    LLM_PROVIDER=openai,    LLM_BASE_URL=https://api.together.xyz/v1
      - Groq:        LLM_PROVIDER=openai,    LLM_BASE_URL=https://api.groq.com/openai/v1

    Nhiệm vụ duy nhất:
      - Nhận messages + tools → gọi LLM → trả về AIMessage
      - KHÔNG chứa logic ReAct loop (việc của agent.py)
      - KHÔNG quản lý session (việc của session_store.py)
    """

    def __init__(self) -> None:
        self._base_url: str | None = settings.llm.base_url or None
        # Cache instances — tránh tạo ChatOpenAI + HTTP client mỗi lần gọi
        self._llm_base: Union[ChatOpenAI, OllamaService, None] = None
        self._llm_with_tools = None
        self._tools_hash: str | None = None

    def _get_llm(self, tools: list[dict] | None = None) -> Union[ChatOpenAI, OllamaService]:
        """
        Khởi tạo ChatOpenAI hoặc OllamaService với config từ settings.
        Cache instance — chỉ tạo mới 1 lần, reuse cho mọi request.
        """
        if self._llm_base is None:
            if settings.llm.provider == "ollama":
                self._llm_base = OllamaService(model=settings.llm.model, base_url=self._base_url)
            else:
                kwargs: dict[str, Any] = dict(
                    model=settings.llm.model,
                    temperature=settings.llm.temperature,
                    max_tokens=settings.llm.max_tokens,
                    api_key=settings.llm.get_api_key(),
                    timeout=60,
                    max_retries=2,
                )
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._llm_base = ChatOpenAI(**kwargs)

        if not tools:
            return self._llm_base

        if settings.llm.provider == "ollama":
            return self._llm_base

        # Cache tools-bound version — dùng content hash thay vì id() để ổn định
        import hashlib, json as _json
        tools_hash = hashlib.md5(
            _json.dumps(tools, sort_keys=True, default=str).encode()
        ).hexdigest()
        if self._llm_with_tools is None or self._tools_hash != tools_hash:
            self._llm_with_tools = self._llm_base.bind_tools(tools)
            self._tools_hash = tools_hash
        return self._llm_with_tools

    # ── Main interface ────────────────────────────────────────

    async def invoke(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AIMessage:
        """
        Gọi LLM 1 lần — agent.py lặp lại hàm này trong ReAct loop.

        Args:
            messages: list[dict] đầy đủ gồm system + history + user message.
                      Format: [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}]
            tools:    list tool schema (OpenAI function format), None nếu không dùng tool.

        Returns:
            AIMessage — agent.py tự đọc .tool_calls hoặc .content
        """
        lc_messages = [_to_lc_message(m) for m in messages]
        llm = self._get_llm(tools)

        # Log user message (last user message in list)
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")[:200]
                break

        tool_names = [t["function"]["name"] for t in (tools or []) if "function" in t]

        logger.info(
            "llm.request",
            model=settings.llm.model,
            provider=self._base_url or "default",
            messages_count=len(lc_messages),
            tools=tool_names or None,
            user_message=user_msg,
        )

        try:
            if settings.llm.provider == "ollama":
                ollama_messages = [_to_ollama_message(m) for m in messages]
                response_dict = await llm.invoke(ollama_messages, tools)
                response = AIMessage(
                    content=response_dict['message']['content'] or "",
                    tool_calls=_normalize_ollama_tool_calls(
                        response_dict['message'].get('tool_calls')
                    ),
                )
            else:
                response: AIMessage = await llm.ainvoke(lc_messages)
            stop_reason = _parse_stop_reason(response)

            # Token usage
            usage = response.response_metadata.get("token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Content preview
            content_preview = ""
            if isinstance(response.content, str):
                content_preview = response.content[:300]
            elif isinstance(response.content, list):
                for block in response.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content_preview = block.get("text", "")[:300]
                        break

            # Tool calls detail
            tc_detail = []
            if response.tool_calls:
                for tc in response.tool_calls:
                    tc_detail.append({
                        "name": tc.get("name"),
                        "args": tc.get("args"),
                    })

            logger.info(
                "llm.response",
                stop_reason=stop_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                tool_calls=tc_detail or None,
                reply_preview=content_preview or None,
            )

            response.response_metadata["stop_reason"] = stop_reason
            return response

        except Exception as e:
            logger.error("llm.error", error=str(e), model=settings.llm.model)
            raise

    async def stream(
        self,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        """
        Stream text response (dùng cho chat UI realtime).
        Không bind tools — chỉ dùng ở lần gọi cuối khi LLM sinh final answer.

        Args:
            messages: list[dict] đầy đủ (system + history + context từ tool results)
        """
        lc_messages = [_to_lc_message(m) for m in messages]
        llm = self._get_llm()

        logger.info(
            "llm.stream_start",
            model=settings.llm.model,
            messages_count=len(lc_messages),
        )

        token_count = 0
        if settings.llm.provider == "ollama":
            async for chunk in llm.stream([_to_ollama_message(m) for m in messages]):
                if chunk:
                    token_count += 1
                    yield chunk
        else:
            async for chunk in llm.astream(lc_messages):
                if chunk.content:
                    token_count += 1
                    yield chunk.content

        logger.info("llm.stream_done", chunks_yielded=token_count)

    async def stream_with_tools(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[Union[str, AIMessage]]:
        """
        Stream LLM response có bind tools.

        Yield:
          - str: từng text token (khi LLM trả text)
          - AIMessage: full response cuối cùng (luôn yield cuối cùng)
            → agent dùng để check tool_calls, extract thinking, v.v.

        Flow:
          1. Stream từng chunk từ LLM
          2. Nếu chunk có text content → yield str ngay (real-time token)
          3. Accumulate toàn bộ chunks → cuối cùng yield AIMessage tổng hợp
        """
        lc_messages = [_to_lc_message(m) for m in messages]
        llm = self._get_llm(tools)

        tool_names = [t["function"]["name"] for t in (tools or []) if "function" in t]
        logger.info(
            "llm.stream_request",
            model=settings.llm.model,
            provider=self._base_url or "default",
            messages_count=len(lc_messages),
            tools=tool_names or None,
        )

        accumulated: AIMessageChunk | None = None
        token_count = 0

        try:
            if settings.llm.provider == "ollama":
                ollama_messages = [_to_ollama_message(m) for m in messages]
                response_dict = await llm.invoke(ollama_messages, tools)
                response_content = response_dict['message']['content'] or ""
                for token in response_content.split():
                    yield token + " "
                final = AIMessage(
                    content=response_content,
                    tool_calls=_normalize_ollama_tool_calls(
                        response_dict['message'].get('tool_calls')
                    ),
                )
                yield final
                return
            
            async for chunk in llm.astream(lc_messages):
                # Accumulate chunks to build full AIMessage
                if accumulated is None:
                    accumulated = chunk
                else:
                    accumulated = accumulated + chunk

                # Yield text token ngay khi có
                if chunk.content:
                    token_count += 1
                    yield chunk.content

            # Build final AIMessage từ accumulated chunks
            if accumulated is not None:
                final = AIMessage(
                    content=accumulated.content or "",
                    tool_calls=accumulated.tool_calls or [],
                    additional_kwargs=accumulated.additional_kwargs or {},
                    response_metadata=accumulated.response_metadata or {},
                )

                # Log completion
                tc_detail = [{"name": tc.get("name"), "args": tc.get("args")} for tc in (final.tool_calls or [])]
                usage = final.response_metadata.get("token_usage", {})
                logger.info(
                    "llm.stream_response",
                    chunks_yielded=token_count,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    tool_calls=tc_detail or None,
                )

                yield final
            else:
                yield AIMessage(content="")

        except Exception as e:
            logger.error("llm_stream_with_tools_error", error=str(e))
            raise

    def extract_tool_calls(self, response: AIMessage) -> list[dict]:
        """
        Trích tool_calls từ AIMessage thành format chuẩn cho agent.py.

        Returns:
            [{"id": "call_abc", "name": "rag_search", "args": {"query": "..."}}]
        """
        if not response.tool_calls:
            return []

        return [
            {
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            }
            for tc in response.tool_calls
        ]

    def get_text(self, response: AIMessage) -> str:
        """Lấy text content từ response (khi stop_reason == END_TURN)."""
        if isinstance(response.content, str):
            return response.content
        parts = [
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts)


# Singleton
llm_client = LLMClient()
