"""
agents/agent.py — ReAct Agent (vòng lặp chính).

Flow: User message → build context → LLM → tool_call? → execute → loop → text → save
Streaming support qua async generator. Max 5 iterations chống infinite loop.
"""

import re
import json as _json
import asyncio
import structlog
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

from config import get_settings
from agents.prompt import build_system_prompt, build_tool_result_message
from agents.tools_registry import get_tool_schemas, execute as execute_tool
from services.llm_client import llm_client
from memory.session_store import session_store

logger = structlog.get_logger(__name__)
settings = get_settings()

MAX_ITERATIONS = 5
MAX_HISTORY_MESSAGES = 40


# ── Data Classes ──────────────────────────────────────────

@dataclass
class AgentRequest:
    user_message: str
    session_id: str
    user_id: str
    tenant_id: str
    role_id: int = 0
    tenant_name: Optional[str] = None
    extra_instructions: Optional[str] = None
    uploaded_files: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    reply: str
    tool_calls_made: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    iterations: int = 0
    latency_ms: float = 0.0
    session_id: str = ""


# ── ReAct Agent ───────────────────────────────────────────

class Agent:

    def __init__(self):
        self.llm = llm_client
        self.session_store = session_store
        self.tool_schemas = get_tool_schemas()

    # ── Shared Setup ──────────────────────────────────

    async def _setup_context(self, request: AgentRequest):
        """Session + history + messages — dùng chung cho run() và run_stream()."""

        session = await self.session_store.get_or_create(
            session_id=request.session_id,
            user_id=request.user_id,
            tenant_id=request.tenant_id,
        )
        system_prompt = build_system_prompt(
            tenant_name=request.tenant_name,
            extra_instructions=request.extra_instructions,
            scoped_files=request.uploaded_files or None,
        )
        history = await self.session_store.get_history(
            session=session, 
            last_n=MAX_HISTORY_MESSAGES,
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": request.user_message})

        return session, messages, len(history)

    def _tool_exec_context(self, request: AgentRequest) -> dict:
        """Build context dict cho tool execution."""

        return {
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "role_id": request.role_id,
            "session_id": request.session_id,
            "uploaded_files": request.uploaded_files or None,
        }

    async def _exec_tools(self, tool_calls: list[dict], context: dict) -> tuple[list, int]:
        """Execute tool calls (parallel nếu >1). Returns (results, latency_ms)."""

        t0 = time.monotonic()
        if len(tool_calls) == 1:
            results = [await execute_tool(
                tool_name=tool_calls[0]["name"],
                arguments=tool_calls[0]["args"],
                context=context,
            )]
        else:
            results = await asyncio.gather(*(
                execute_tool(tool_name=tc["name"], arguments=tc["args"], context=context)
                for tc in tool_calls
            ))

        return results, int((time.monotonic() - t0) * 1000)

    @staticmethod
    def _tool_log_summary(tc_name: str, result: dict) -> dict:
        """Build log summary dict cho 1 tool result."""

        summary: dict = {
            "status": result.get("status", "ok"), 
            "error": result.get("error")
            }

        if tc_name == "rag_search":
            summary["results_count"] = len(result.get("results", []))
            summary["cached"] = result.get("cached", False)
        elif tc_name == "erp_db_query":
            summary["row_count"] = result.get("row_count", 0)

        return summary

    async def _save_messages(self, session, request: AgentRequest, reply: str, messages: list[dict]):
        """Save user + assistant messages với tool context."""

        tool_context = self._build_tool_context(messages)
        await self.session_store.add_message(
            session=session, role="user", content=request.user_message,
        )
        if reply:
            await self.session_store.add_message(
                session=session, role="assistant", content=reply,
                tool_context=tool_context or None,
            )

    # ── run() — Non-streaming (LLM vẫn dùng stream internally) ──

    async def run(self, request: AgentRequest) -> AgentResponse:
        start_time = time.monotonic()
        log = logger.bind(
            session_id=request.session_id, tenant_id=request.tenant_id,
            user_id=request.user_id, role_id=request.role_id,
        )
        log.info("═" * 60)
        log.info("agent.start", message=request.user_message[:200],
                 scoped_files=request.uploaded_files or None)

        session, messages, hist_len = await self._setup_context(request)
        log.info("agent.context", history_messages=hist_len, total_messages=len(messages))

        context = self._tool_exec_context(request)
        tool_calls_made: list[dict] = []
        rag_results: list[dict] = []
        iteration = 0
        reply = ""

        while iteration < MAX_ITERATIONS:
            iteration += 1
            log.info("agent.iteration", iteration=iteration, messages_count=len(messages))

            try:
                # Stream internally — consume all tokens, collect response
                streamed_tokens: list[str] = []
                response = None
                async for item in self.llm.stream_with_tools(messages, tools=self.tool_schemas):
                    if isinstance(item, str):
                        streamed_tokens.append(item)
                    else:
                        response = item

                tool_calls = self.llm.extract_tool_calls(response)

                if not tool_calls:
                    raw_text = ("".join(streamed_tokens)
                                if streamed_tokens
                                else self._extract_text(response))
                    _, reply = self._extract_thinking(raw_text)

                    log.info("agent.final_reply", iteration=iteration, reply_len=len(reply), reply_preview=reply[:300])
                    break

                log.info("agent.tool_calls", iteration=iteration, tools=[tc["name"] for tc in tool_calls])
                messages.append(self._response_to_message(response))

                results_list, tool_ms = await self._exec_tools(tool_calls, context)

                for tc, result in zip(tool_calls, results_list):
                    log.info("agent.tool_exec", tool=tc["name"], args=tc["args"])
                    summary = self._tool_log_summary(tc["name"], result)
                    log.info("agent.tool_result", tool=tc["name"], latency_ms=tool_ms, **summary)

                    tool_calls_made.append({
                        "tool": tc["name"],
                        "arguments": tc["args"],
                        "result_status": result.get("status", "ok"),
                    })
                    if tc["name"] == "rag_search" and "results" in result:
                        rag_results.extend(result["results"])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": build_tool_result_message(tc["name"], result),
                    })

            except Exception as e:
                log.error("agent.iteration_error", iteration=iteration, error=str(e))
                reply = "Xin lỗi, đã xảy ra lỗi khi xử lý. Vui lòng thử lại."
                break
        else:
            log.warning("agent.max_iterations", iterations=MAX_ITERATIONS)
            reply = (
                "Xin lỗi, tôi đã thử nhiều cách nhưng không thể hoàn thành yêu cầu. "
                "Vui lòng thử lại hoặc diễn đạt câu hỏi khác."
            )

        # Citations + Save
        citations = self._extract_citations(rag_results)
        if citations:
            reply += self._build_sources_text(citations)
            log.info("agent.citations", sources=[c["source"] for c in citations])

        await self._save_messages(session, request, reply, messages)

        latency_ms = (time.monotonic() - start_time) * 1000
        log.info("agent.done", iterations=iteration,
                 tools_called=[t["tool"] for t in tool_calls_made],
                 latency_ms=f"{latency_ms:.0f}", reply_len=len(reply))

        return AgentResponse(
            reply=reply, tool_calls_made=tool_calls_made, citations=citations,
            iterations=iteration, latency_ms=latency_ms, session_id=request.session_id,
        )

    # ── run_stream() — Streaming ──────────────────────

    async def run_stream(self, request: AgentRequest) -> AsyncGenerator[str, None]:
        """
        Streaming version — yield JSON events:
          thinking, reasoning, tool (running/done), token, citations, done
        """
        def _j(obj: dict) -> str:
            return _json.dumps(obj, ensure_ascii=False)

        start_ts = time.monotonic()
        log = logger.bind(
            session_id=request.session_id, tenant_id=request.tenant_id,
            user_id=request.user_id, role_id=request.role_id, mode="stream",
        )
        log.info("═" * 60)
        log.info("agent.stream_start", message=request.user_message[:200],
                 scoped_files=request.uploaded_files or None)

        t0 = time.monotonic()
        session, messages, hist_len = await self._setup_context(request)
        setup_ms = int((time.monotonic() - t0) * 1000)
        log.info("agent.stream_context", history_messages=hist_len,
                 total_messages=len(messages), setup_ms=setup_ms)

        context = self._tool_exec_context(request)
        iteration = 0
        full_reply = ""
        rag_results: list[dict] = []

        while iteration < MAX_ITERATIONS:
            iteration += 1
            log.info("agent.stream_iteration", iteration=iteration,
                     messages_count=len(messages))

            try:
                yield _j({"type": "thinking", "content":
                          "Đang phân tích câu hỏi…" if iteration == 1
                          else f"Đang tiếp tục xử lý (bước {iteration})…"})

                llm_start = time.monotonic()
                streamed_tokens: list[str] = []
                response = None
                ttft = None

                async for item in self.llm.stream_with_tools(
                    messages, tools=self.tool_schemas
                ):
                    if isinstance(item, str):
                        if ttft is None:
                            ttft = int((time.monotonic() - llm_start) * 1000)
                        streamed_tokens.append(item)
                        yield _j({"type": "token", "content": item})
                    else:
                        response = item

                llm_ms = int((time.monotonic() - llm_start) * 1000)
                log.info("agent.stream_llm_timing", iteration=iteration,
                         ttft_ms=ttft, llm_total_ms=llm_ms,
                         streamed_chunks=len(streamed_tokens),
                         has_tool_calls=bool(
                             self.llm.extract_tool_calls(response)
                             if response else False
                         ))

                # Extract model thinking (reasoning_content hoặc <think> tag)
                model_thinking = ""
                if response and hasattr(response, "additional_kwargs"):
                    model_thinking = (
                        response.additional_kwargs.get("reasoning_content", "")
                        or response.additional_kwargs.get("reasoning", "")
                        or ""
                    )
                if not model_thinking and streamed_tokens:
                    model_thinking, _ = self._extract_thinking(
                        "".join(streamed_tokens)
                    )
                if model_thinking:
                    yield _j({"type": "reasoning", "content": model_thinking})

                tool_calls = self.llm.extract_tool_calls(response)

                # LLM trả text → done
                if not tool_calls:
                    raw = ("".join(streamed_tokens)
                           if streamed_tokens
                           else self._extract_text(response))
                    _, full_reply = self._extract_thinking(raw)
                    log.info("agent.stream_final", iteration=iteration,
                             reply_len=len(full_reply),
                             reply_preview=full_reply[:300])
                    break

                # LLM gọi tool(s) → execute + loop
                log.info("agent.stream_tool_calls", iteration=iteration,
                         tools=[tc["name"] for tc in tool_calls])
                messages.append(self._response_to_message(response))

                for tc in tool_calls:
                    log.info("agent.stream_tool_exec",
                             tool=tc["name"], args=tc["args"])
                    yield _j({"type": "tool", "name": tc["name"],
                              "status": "running"})

                results_list, tool_ms = await self._exec_tools(
                    tool_calls, context
                )

                for tc, result in zip(tool_calls, results_list):
                    summary = self._tool_log_summary(tc["name"], result)
                    log.info("agent.stream_tool_result", tool=tc["name"],
                             latency_ms=tool_ms, **summary)

                    if tc["name"] == "rag_search" and "results" in result:
                        rag_results.extend(result["results"])

                    yield _j({"type": "tool", "name": tc["name"],
                              "args": tc["args"], "result": result,
                              "latency_ms": tool_ms})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": build_tool_result_message(
                            tc["name"], result
                        ),
                    })

            except Exception as e:
                log.error("agent.stream_iteration_error",
                          iteration=iteration, error=str(e))
                full_reply = "Xin lỗi, đã xảy ra lỗi khi xử lý. Vui lòng thử lại."
                yield _j({"type": "error", "content": full_reply})
                break
        else:
            full_reply = "Xin lỗi, tôi không thể hoàn thành yêu cầu. Vui lòng thử lại."
            yield _j({"type": "token", "content": full_reply})

        # Citations
        citations = self._extract_citations(rag_results)
        if citations:
            full_reply += self._build_sources_text(citations)
            log.info("agent.stream_citations",
                     sources=[c["source"] for c in citations])
            yield _j({"type": "citations", "citations": citations})

        # Done
        total_ms = int((time.monotonic() - start_ts) * 1000)
        log.info("agent.stream_done", iterations=iteration,
                 latency_ms=total_ms, reply_len=len(full_reply))
        yield _j({"type": "done", "session_id": request.session_id,
                  "latency_ms": total_ms})

        # Save
        await self._save_messages(session, request, full_reply, messages)

    # ── Tool Context for History ──────────────────────

    @staticmethod
    def _build_tool_context(messages: list[dict], max_per_tool: int = 500) -> str:
        """Condensed tool context cho history — enables follow-up questions."""

        SKIP_PREFIXES = (
            "Tool `", "Query thành công nhưng không",
            "Query thành công nhưng trả về 0", "Không tìm thấy tài liệu",
        )

        # Map tool_call_id → {name, args}
        tc_map: dict[str, dict] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    tc_map[tc.get("id", "")] = {
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    }

        tool_parts = []
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not content or any(content.startswith(p) for p in SKIP_PREFIXES):
                continue

            tc_info = tc_map.get(msg.get("tool_call_id", ""), {})
            name = tc_info.get("name", "unknown")
            args = tc_info.get("args", {})

            # Format args ngắn gọn
            if name == "erp_db_query":
                args_str = args.get("query", "")[:200]
            elif name == "rag_search":
                args_str = args.get("query", "")[:100]
            else:
                args_str = _json.dumps(args, ensure_ascii=False)[:150]

            # Truncate result — giữ header + first rows
            if len(content) > max_per_tool:
                lines = content.split("\n")
                kept: list[str] = []
                length = 0
                for line in lines:
                    if length + len(line) > max_per_tool:
                        kept.append("… (rút gọn)")
                        break
                    kept.append(line)
                    length += len(line) + 1
                content = "\n".join(kept)

            tool_parts.append(f"[{name}] {args_str}\n{content}")

        if not tool_parts:
            return ""
        result = "\n---\n".join(tool_parts)
        return result[:1500] + "\n… (rút gọn)" if len(result) > 1500 else result

    # ── Citations ─────────────────────────────────────

    @staticmethod
    def _extract_citations(rag_results: list[dict]) -> list[dict]:
        """Gom RAG results theo filename, sort by score."""

        if not rag_results:
            return []

        grouped: dict[str, dict] = {}
        for doc in rag_results:
            source = doc.get("metadata", {}).get("filename", "N/A")
            score = doc.get("score", 0.0)
            chunk = {"content": doc.get("content", "")}

            if source not in grouped:
                grouped[source] = {
                    "source": source, "_score": round(score, 4),
                    "chunks": [chunk],
                }
            else:
                grouped[source]["chunks"].append(chunk)
                if score > grouped[source]["_score"]:
                    grouped[source]["_score"] = round(score, 4)

        citations = sorted(
            grouped.values(), key=lambda c: c["_score"], reverse=True
        )
        for cite in citations:
            cite.pop("_score", None)
        return citations

    @staticmethod
    def _build_sources_text(citations: list[dict]) -> str:
        if not citations:
            return ""
        lines = ["\n\n---", "**📚 Tài liệu tham khảo:**\n"]
        for idx, cite in enumerate(citations, 1):
            n = len(cite.get("chunks", []))
            suffix = f" ({n} đoạn trích dẫn)" if n > 1 else ""
            lines.append(f"{idx}. **{cite['source']}**{suffix}")
        return "\n".join(lines)

    # ── Private Helpers ───────────────────────────────

    @staticmethod
    def _extract_text(response) -> str:
        """Lấy text content từ LLM response (AIMessage)."""

        if hasattr(response, "content"):
            c = response.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "".join(
                    p if isinstance(p, str) else p.get("text", "")
                    for p in c if isinstance(p, (str, dict))
                )
        return str(response)

    @staticmethod
    def _extract_thinking(text: str) -> tuple[str, str]:
        """Parse <think>...</think> tags → (thinking, clean_content)."""

        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            return (match.group(1).strip(),
                    (text[:match.start()] + text[match.end():]).strip())
        return "", text

    @staticmethod
    def _response_to_message(response) -> dict:
        """Convert LangChain AIMessage → OpenAI message format."""
        
        msg: dict = {"role": "assistant"}

        text = ""
        if hasattr(response, "content"):
            if isinstance(response.content, str):
                text = response.content
            elif isinstance(response.content, list):
                for part in response.content:
                    if isinstance(part, str):
                        text += part
                    elif isinstance(part, dict) and part.get("type") == "text":
                        text += part.get("text", "")

        msg["content"] = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL
        ).strip()

        if hasattr(response, "tool_calls") and response.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.get("id", ""), "name": tc.get("name", ""),
                 "args": tc.get("args", {}), "type": "tool_call"}
                for tc in response.tool_calls
            ]
        return msg


agent = Agent()
