"""
OpenTelemetry Middleware for LangChain/LangGraph

This module provides OpenTelemetry instrumentation for:
- LLM calls (with token usage tracking)
- Tool calls
- Graph execution tracing
- Agent middleware for create_agent

Usage with callback handler (for graphs):
    from otel_middleware import setup_otel_tracing, OtelCallbackHandler
    
    # Initialize tracing at app startup
    setup_otel_tracing(service_name="my-langchain-app")
    
    # Use the callback handler with your LLM/graph
    handler = OtelCallbackHandler()
    llm.invoke(messages, config={"callbacks": [handler]})

Usage with agent middleware (for create_agent):
    from otel_middleware import setup_otel_tracing, OtelMiddleware
    from langchain.agents import create_agent
    
    # Initialize tracing at app startup
    setup_otel_tracing(service_name="my-agent-app")
    
    # Use the middleware with create_agent
    agent = create_agent(
        model=llm,
        tools=tools,
        middleware=[OtelMiddleware(trace_content=True)]
    )
"""

import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Awaitable
from uuid import UUID
from contextlib import contextmanager
from dataclasses import dataclass

from opentelemetry import trace, metrics, context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.context import Context

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import BaseTool

# Import for AgentMiddleware
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.types import ToolCallRequest
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Global tracer and meter
_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None

# Metrics
_llm_call_counter: Optional[metrics.Counter] = None
_llm_token_counter: Optional[metrics.Counter] = None
_llm_duration_histogram: Optional[metrics.Histogram] = None
_tool_call_counter: Optional[metrics.Counter] = None
_tool_duration_histogram: Optional[metrics.Histogram] = None


def setup_otel_tracing(
    service_name: str = "langchain-agent",
    otlp_endpoint: str = "localhost:4317",
    enable_console_export: bool = False,
) -> None:
    """
    Initialize OpenTelemetry tracing and metrics.
    
    Args:
        service_name: Name of the service for tracing
        otlp_endpoint: OTLP gRPC endpoint (host:port)
        enable_console_export: If True, also export to console for debugging
    """
    global _tracer, _meter
    global _llm_call_counter, _llm_token_counter, _llm_duration_histogram
    global _tool_call_counter, _tool_duration_histogram
    
    # Create resource with service name
    resource = Resource.create({SERVICE_NAME: service_name})
    
    # Setup Tracer Provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Add OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=True,
    )
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Optionally add console exporter for debugging
    if enable_console_export:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
        tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer(__name__, "1.0.0")
    
    # Setup Meter Provider
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=otlp_endpoint,
            insecure=True,
        ),
        export_interval_millis=10000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    _meter = metrics.get_meter(__name__, "1.0.0")
    
    # Create metrics instruments
    _llm_call_counter = _meter.create_counter(
        name="llm.calls",
        description="Number of LLM calls",
        unit="1",
    )
    
    _llm_token_counter = _meter.create_counter(
        name="llm.tokens",
        description="Number of tokens used",
        unit="tokens",
    )
    
    _llm_duration_histogram = _meter.create_histogram(
        name="llm.duration",
        description="Duration of LLM calls",
        unit="ms",
    )
    
    _tool_call_counter = _meter.create_counter(
        name="tool.calls",
        description="Number of tool calls",
        unit="1",
    )
    
    _tool_duration_histogram = _meter.create_histogram(
        name="tool.duration",
        description="Duration of tool calls",
        unit="ms",
    )
    
    logger.info(f"OpenTelemetry tracing initialized for service: {service_name}")


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        # Return a no-op tracer if not initialized
        return trace.get_tracer(__name__)
    return _tracer


@contextmanager
def trace_llm_call(
    model_name: str,
    provider: str = "unknown",
    **attributes,
):
    """Context manager for tracing LLM calls."""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        f"llm.{provider}.{model_name}",
        kind=SpanKind.CLIENT,
    ) as span:
        span.set_attribute("llm.model", model_name)
        span.set_attribute("llm.provider", provider)
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("gen_ai.request.model", model_name)
        
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@contextmanager
def trace_tool_call(
    tool_name: str,
    **attributes,
):
    """Context manager for tracing tool calls."""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        f"tool.{tool_name}",
        kind=SpanKind.INTERNAL,
    ) as span:
        span.set_attribute("tool.name", tool_name)
        
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


class OtelCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for OpenTelemetry instrumentation.
    
    Tracks:
    - LLM calls with token usage
    - Tool/function calls
    - Chain execution
    """
    
    def __init__(self, trace_content: bool = False):
        """
        Initialize the callback handler.
        
        Args:
            trace_content: If True, include prompt/response content in traces
                          (be careful with sensitive data)
        """
        super().__init__()
        self.trace_content = trace_content
        self._spans: Dict[UUID, trace.Span] = {}
        self._start_times: Dict[UUID, float] = {}
        self._token_usage: Dict[UUID, Dict[str, int]] = {}
    
    @property
    def tracer(self) -> trace.Tracer:
        return get_tracer()
    
    # ==================== LLM Callbacks ====================
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""
        model_name = serialized.get("kwargs", {}).get("model", "unknown")
        model_name = serialized.get("kwargs", {}).get("model_name", model_name)
        provider = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
        
        # Start span
        span = self.tracer.start_span(
            f"llm.call",
            kind=SpanKind.CLIENT,
        )
        
        span.set_attribute("llm.model", model_name)
        span.set_attribute("llm.provider", provider)
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("gen_ai.request.model", model_name)
        span.set_attribute("llm.prompt_count", len(prompts))
        
        if tags:
            span.set_attribute("llm.tags", tags)
        
        if self.trace_content and prompts:
            # Only store first prompt to avoid huge traces
            span.set_attribute("gen_ai.prompt", prompts[0][:1000])
        
        self._spans[run_id] = span
        self._start_times[run_id] = time.time()
    
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chat model starts running."""
        model_name = serialized.get("kwargs", {}).get("model", "unknown")
        model_name = serialized.get("kwargs", {}).get("model_name", model_name)
        provider = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
        
        # Count messages
        total_messages = sum(len(msg_list) for msg_list in messages)
        
        # Start span
        span = self.tracer.start_span(
            f"llm.chat",
            kind=SpanKind.CLIENT,
        )
        
        span.set_attribute("llm.model", model_name)
        span.set_attribute("llm.provider", provider)
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("gen_ai.request.model", model_name)
        span.set_attribute("llm.message_count", total_messages)
        span.set_attribute("llm.type", "chat")
        
        if tags:
            span.set_attribute("llm.tags", tags)
        
        if self.trace_content and messages:
            # Store last message content
            last_msg = messages[0][-1] if messages and messages[0] else None
            if last_msg:
                content = str(last_msg.content)[:1000]
                span.set_attribute("gen_ai.prompt", content)
        
        self._spans[run_id] = span
        self._start_times[run_id] = time.time()
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends running."""
        span = self._spans.pop(run_id, None)
        start_time = self._start_times.pop(run_id, None)
        
        if span is None:
            return
        
        duration_ms = (time.time() - start_time) * 1000 if start_time else 0
        
        # Extract token usage
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            if not token_usage:
                token_usage = response.llm_output.get("usage", {})
        
        # Also check generations for usage_metadata (newer LangChain)
        if not token_usage and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                        usage = gen.message.usage_metadata
                        if usage:
                            token_usage = {
                                "prompt_tokens": usage.get("input_tokens", 0),
                                "completion_tokens": usage.get("output_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                            }
                            break
        
        prompt_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
        
        # Set span attributes
        span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
        span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
        span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
        span.set_attribute("llm.duration_ms", duration_ms)
        
        if self.trace_content and response.generations:
            # Store first generation content
            first_gen = response.generations[0][0] if response.generations and response.generations[0] else None
            if first_gen:
                content = str(first_gen.text)[:1000] if hasattr(first_gen, "text") else ""
                if not content and hasattr(first_gen, "message"):
                    content = str(first_gen.message.content)[:1000]
                span.set_attribute("gen_ai.completion", content)
        
        span.set_status(Status(StatusCode.OK))
        span.end()
        
        # Record metrics
        model = span.attributes.get("llm.model", "unknown") if hasattr(span, "attributes") else "unknown"
        provider = span.attributes.get("llm.provider", "unknown") if hasattr(span, "attributes") else "unknown"
        
        labels = {"model": str(model), "provider": str(provider)}
        
        if _llm_call_counter:
            _llm_call_counter.add(1, labels)
        
        if _llm_token_counter:
            _llm_token_counter.add(prompt_tokens, {**labels, "type": "prompt"})
            _llm_token_counter.add(completion_tokens, {**labels, "type": "completion"})
        
        if _llm_duration_histogram:
            _llm_duration_histogram.record(duration_ms, labels)
    
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        span = self._spans.pop(run_id, None)
        self._start_times.pop(run_id, None)
        
        if span is None:
            return
        
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
        span.end()
    
    # ==================== Tool Callbacks ====================
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts running."""
        tool_name = serialized.get("name", "unknown_tool")
        
        span = self.tracer.start_span(
            f"tool.{tool_name}",
            kind=SpanKind.INTERNAL,
        )
        
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.description", serialized.get("description", "")[:200])
        
        if tags:
            span.set_attribute("tool.tags", tags)
        
        if self.trace_content:
            span.set_attribute("tool.input", input_str[:1000])
        
        self._spans[run_id] = span
        self._start_times[run_id] = time.time()
    
    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends running."""
        span = self._spans.pop(run_id, None)
        start_time = self._start_times.pop(run_id, None)
        
        if span is None:
            return
        
        duration_ms = (time.time() - start_time) * 1000 if start_time else 0
        
        span.set_attribute("tool.duration_ms", duration_ms)
        
        if self.trace_content:
            span.set_attribute("tool.output", str(output)[:1000])
        
        span.set_status(Status(StatusCode.OK))
        span.end()
        
        # Record metrics
        tool_name = span.attributes.get("tool.name", "unknown") if hasattr(span, "attributes") else "unknown"
        
        if _tool_call_counter:
            _tool_call_counter.add(1, {"tool": str(tool_name)})
        
        if _tool_duration_histogram:
            _tool_duration_histogram.record(duration_ms, {"tool": str(tool_name)})
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        span = self._spans.pop(run_id, None)
        self._start_times.pop(run_id, None)
        
        if span is None:
            return
        
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
        span.end()
    
    # ==================== Chain Callbacks ====================
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts running."""
        chain_name = serialized.get("name", serialized.get("id", ["chain"])[-1])
        
        span = self.tracer.start_span(
            f"chain.{chain_name}",
            kind=SpanKind.INTERNAL,
        )
        
        span.set_attribute("chain.name", chain_name)
        
        if tags:
            span.set_attribute("chain.tags", tags)
        
        self._spans[run_id] = span
        self._start_times[run_id] = time.time()
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain ends running."""
        span = self._spans.pop(run_id, None)
        start_time = self._start_times.pop(run_id, None)
        
        if span is None:
            return
        
        duration_ms = (time.time() - start_time) * 1000 if start_time else 0
        span.set_attribute("chain.duration_ms", duration_ms)
        
        span.set_status(Status(StatusCode.OK))
        span.end()
    
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        span = self._spans.pop(run_id, None)
        self._start_times.pop(run_id, None)
        
        if span is None:
            return
        
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)
        span.end()
    
    # ==================== Agent Callbacks ====================
    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""
        tracer = get_tracer()
        with tracer.start_as_current_span(
            f"agent.action.{action.tool}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("agent.tool", action.tool)
            span.set_attribute("agent.log", action.log[:500] if action.log else "")
            
            if self.trace_content:
                span.set_attribute("agent.tool_input", str(action.tool_input)[:1000])
    
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "agent.finish",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("agent.log", finish.log[:500] if finish.log else "")
            
            if self.trace_content:
                span.set_attribute("agent.return_values", str(finish.return_values)[:1000])


def create_traced_config(
    trace_content: bool = False,
    additional_callbacks: Optional[List[BaseCallbackHandler]] = None,
) -> Dict[str, Any]:
    """
    Create a config dict with OTel callback handler for use with LangChain.
    
    Args:
        trace_content: Whether to include content in traces
        additional_callbacks: Additional callback handlers to include
    
    Returns:
        Config dict with callbacks configured
    
    Example:
        config = create_traced_config()
        result = llm.invoke(messages, config=config)
    """
    callbacks = [OtelCallbackHandler(trace_content=trace_content)]
    if additional_callbacks:
        callbacks.extend(additional_callbacks)
    
    return {"callbacks": callbacks}


@contextmanager
def trace_agent_invocation(
    agent_name: str = "agent",
    trace_content: bool = False,
    **attributes,
):
    """
    Context manager for tracing an agent invocation.
    
    Use this to wrap agent.invoke() calls to create a root span that will
    contain all nested operations (LLM calls, tool calls, sub-agent calls).
    
    Args:
        agent_name: Name of the agent for the span
        trace_content: If True, include content in traces
        **attributes: Additional span attributes
    
    Example:
        with trace_agent_invocation("main_agent"):
            result = agent.invoke({"messages": [...]})
    
    This creates a trace hierarchy like:
        agent.main_agent (root)
        ├── llm.ChatOpenAI.gpt-4o-mini
        ├── tool.ask_weather_agent
        │   ├── llm.ChatOpenAI.gpt-4o-mini
        │   ├── tool.get_weather
        │   └── llm.ChatOpenAI.gpt-4o-mini
        └── llm.ChatOpenAI.gpt-4o-mini
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        f"agent.{agent_name}",
        kind=SpanKind.INTERNAL,
    ) as span:
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("agent.type", "invocation")
        
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        
        start_time = time.time()
        
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            span.set_attribute("agent.duration_ms", duration_ms)


# ============================================================
# Agent Middleware for create_agent
# ============================================================


class OtelMiddleware(AgentMiddleware):
    """
    OpenTelemetry middleware for LangChain create_agent.
    
    This middleware traces:
    - Agent execution lifecycle (before/after agent)
    - Model calls (before/after model, wrap_model_call)
    - Tool calls (wrap_tool_call)
    
    Attributes:
        trace_content: If True, include prompt/response/tool content in traces
        service_name: Optional service name for span attributes
    
    Example:
        from langchain.agents import create_agent
        from otel_middleware import setup_otel_tracing, OtelMiddleware
        
        setup_otel_tracing(service_name="my-agent")
        
        agent = create_agent(
            model=llm,
            tools=tools,
            middleware=[OtelMiddleware(trace_content=True)]
        )
    """
    
    def __init__(
        self,
        trace_content: bool = False,
        service_name: Optional[str] = None,
    ):
        """
        Initialize the OTel middleware.
        
        Args:
            trace_content: If True, include content in traces (careful with sensitive data)
            service_name: Optional service name for span attributes
        """
        self.trace_content = trace_content
        self.service_name = service_name
        self._agent_spans: Dict[str, trace.Span] = {}
        self._agent_start_times: Dict[str, float] = {}
        self._context_tokens: Dict[str, object] = {}  # Store context tokens for proper attach/detach
        self.tools: list[BaseTool] = []
    
    @property
    def tracer(self) -> trace.Tracer:
        """Get the global tracer instance."""
        return get_tracer()
    
    def _extract_model_info(self, request: ModelRequest) -> Dict[str, Any]:
        """Extract model information from request."""
        model = request.model
        model_name = getattr(model, "model", None) or getattr(model, "model_name", "unknown")
        provider = model.__class__.__name__
        return {
            "model_name": model_name,
            "provider": provider,
        }
    
    # ==================== Agent Lifecycle ====================
    
    def before_agent(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        """Called before the agent starts execution.
        
        Creates a span for the agent execution and attaches it to the OpenTelemetry
        context so that all child spans (model calls, tool calls, sub-agent calls)
        are properly nested under this span.
        """
        run_id = str(id(runtime))
        
        # Start a new span. If there's a current span in context (e.g., from parent agent's
        # tool call), this span will automatically be a child of it.
        span_name = f"agent.{self.service_name}" if self.service_name else "agent.execution"
        span = self.tracer.start_span(
            span_name,
            kind=SpanKind.INTERNAL,
        )
        
        span.set_attribute("agent.type", "create_agent")
        if self.service_name:
            span.set_attribute("service.name", self.service_name)
            span.set_attribute("agent.name", self.service_name)
        
        # Track message count
        messages = state.get("messages", [])
        if messages:
            span.set_attribute("agent.input_message_count", len(messages))
            if self.trace_content and messages:
                last_msg = messages[-1]
                content = str(getattr(last_msg, "content", str(last_msg)))[:500]
                span.set_attribute("agent.input_message", content)
        
        # CRITICAL: Attach the span to the OpenTelemetry context
        # This makes it the current span, so all child operations (model calls,
        # tool calls, sub-agent invocations) will be properly nested under it
        new_context = trace.set_span_in_context(span)
        token = otel_context.attach(new_context)
        
        self._agent_spans[run_id] = span
        self._agent_start_times[run_id] = time.time()
        self._context_tokens[run_id] = token
        
        return None
    
    def after_agent(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        """Called after the agent completes execution.
        
        Ends the agent execution span and detaches it from the OpenTelemetry context,
        restoring the previous context.
        """
        run_id = str(id(runtime))
        
        span = self._agent_spans.pop(run_id, None)
        start_time = self._agent_start_times.pop(run_id, None)
        token = self._context_tokens.pop(run_id, None)
        
        if span:
            duration_ms = (time.time() - start_time) * 1000 if start_time else 0
            span.set_attribute("agent.duration_ms", duration_ms)
            
            # Track output message count
            messages = state.get("messages", [])
            if messages:
                span.set_attribute("agent.output_message_count", len(messages))
            
            span.set_status(Status(StatusCode.OK))
            span.end()
        
        # CRITICAL: Detach the context to restore the previous span context
        # This ensures proper context cleanup and allows parent spans to continue correctly
        if token is not None:
            otel_context.detach(token)
        
        return None
    
    # ==================== Model Calls ====================
    
    def before_model(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        """Called before the model is invoked."""
        # We use wrap_model_call for more detailed tracing
        return None
    
    def after_model(self, state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
        """Called after the model returns."""
        # We use wrap_model_call for more detailed tracing
        return None
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Wrap model call with OpenTelemetry tracing."""
        model_info = self._extract_model_info(request)
        model_name = model_info["model_name"]
        provider = model_info["provider"]
        
        with self.tracer.start_as_current_span(
            f"llm.{provider}.{model_name}",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute("llm.model", model_name)
            span.set_attribute("llm.provider", provider)
            span.set_attribute("gen_ai.system", provider)
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("llm.message_count", len(request.messages))
            
            if request.system_prompt:
                span.set_attribute("llm.has_system_prompt", True)
            
            if request.tools:
                span.set_attribute("llm.tool_count", len(request.tools))
                tool_names = [
                    t.name if isinstance(t, BaseTool) else t.get("name", "unknown")
                    for t in request.tools
                ]
                span.set_attribute("llm.tools", tool_names)
            
            if self.trace_content and request.messages:
                last_msg = request.messages[-1]
                content = str(getattr(last_msg, "content", str(last_msg)))[:1000]
                span.set_attribute("gen_ai.prompt", content)
            
            start_time = time.time()
            
            try:
                response = handler(request)
                
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
                
                # Extract token usage from response
                if response.result:
                    for msg in response.result:
                        if isinstance(msg, AIMessage):
                            usage = getattr(msg, "usage_metadata", None)
                            if usage:
                                prompt_tokens = usage.get("input_tokens", 0)
                                completion_tokens = usage.get("output_tokens", 0)
                                total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
                                
                                span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
                                span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
                                span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
                                
                                # Record metrics
                                labels = {"model": str(model_name), "provider": str(provider)}
                                if _llm_call_counter:
                                    _llm_call_counter.add(1, labels)
                                if _llm_token_counter:
                                    _llm_token_counter.add(prompt_tokens, {**labels, "type": "prompt"})
                                    _llm_token_counter.add(completion_tokens, {**labels, "type": "completion"})
                                if _llm_duration_histogram:
                                    _llm_duration_histogram.record(duration_ms, labels)
                            
                            # Check for tool calls
                            tool_calls = getattr(msg, "tool_calls", None)
                            if tool_calls:
                                span.set_attribute("llm.tool_call_count", len(tool_calls))
                                tool_call_names = [tc.get("name", "unknown") for tc in tool_calls]
                                span.set_attribute("llm.tool_calls", tool_call_names)
                            
                            if self.trace_content:
                                content = str(msg.content)[:1000] if msg.content else ""
                                span.set_attribute("gen_ai.completion", content)
                
                span.set_status(Status(StatusCode.OK))
                return response
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        model_info = self._extract_model_info(request)
        model_name = model_info["model_name"]
        provider = model_info["provider"]
        
        with self.tracer.start_as_current_span(
            f"llm.{provider}.{model_name}",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute("llm.model", model_name)
            span.set_attribute("llm.provider", provider)
            span.set_attribute("gen_ai.system", provider)
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("llm.message_count", len(request.messages))
            
            if request.system_prompt:
                span.set_attribute("llm.has_system_prompt", True)
            
            if request.tools:
                span.set_attribute("llm.tool_count", len(request.tools))
                tool_names = [
                    t.name if isinstance(t, BaseTool) else t.get("name", "unknown")
                    for t in request.tools
                ]
                span.set_attribute("llm.tools", tool_names)
            
            if self.trace_content and request.messages:
                last_msg = request.messages[-1]
                content = str(getattr(last_msg, "content", str(last_msg)))[:1000]
                span.set_attribute("gen_ai.prompt", content)
            
            start_time = time.time()
            
            try:
                response = await handler(request)
                
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("llm.duration_ms", duration_ms)
                
                # Extract token usage from response
                if response.result:
                    for msg in response.result:
                        if isinstance(msg, AIMessage):
                            usage = getattr(msg, "usage_metadata", None)
                            if usage:
                                prompt_tokens = usage.get("input_tokens", 0)
                                completion_tokens = usage.get("output_tokens", 0)
                                total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
                                
                                span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
                                span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
                                span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
                                
                                # Record metrics
                                labels = {"model": str(model_name), "provider": str(provider)}
                                if _llm_call_counter:
                                    _llm_call_counter.add(1, labels)
                                if _llm_token_counter:
                                    _llm_token_counter.add(prompt_tokens, {**labels, "type": "prompt"})
                                    _llm_token_counter.add(completion_tokens, {**labels, "type": "completion"})
                                if _llm_duration_histogram:
                                    _llm_duration_histogram.record(duration_ms, labels)
                            
                            # Check for tool calls
                            tool_calls = getattr(msg, "tool_calls", None)
                            if tool_calls:
                                span.set_attribute("llm.tool_call_count", len(tool_calls))
                                tool_call_names = [tc.get("name", "unknown") for tc in tool_calls]
                                span.set_attribute("llm.tool_calls", tool_call_names)
                            
                            if self.trace_content:
                                content = str(msg.content)[:1000] if msg.content else ""
                                span.set_attribute("gen_ai.completion", content)
                
                span.set_status(Status(StatusCode.OK))
                return response
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    # ==================== Tool Calls ====================
    
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Wrap tool call with OpenTelemetry tracing."""
        tool_name = request.tool_call.get("name", "unknown")
        tool_id = request.tool_call.get("id", "")
        
        with self.tracer.start_as_current_span(
            f"tool.{tool_name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.call_id", tool_id)
            
            if request.tool:
                span.set_attribute("tool.description", (request.tool.description or "")[:200])
            
            if self.trace_content:
                args = request.tool_call.get("args", {})
                span.set_attribute("tool.input", str(args)[:1000])
            
            start_time = time.time()
            
            try:
                result = handler(request)
                
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("tool.duration_ms", duration_ms)
                
                if self.trace_content and isinstance(result, ToolMessage):
                    span.set_attribute("tool.output", str(result.content)[:1000])
                
                # Check for error status
                if isinstance(result, ToolMessage):
                    status = getattr(result, "status", None)
                    if status == "error":
                        span.set_status(Status(StatusCode.ERROR, "Tool returned error"))
                    else:
                        span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status(Status(StatusCode.OK))
                
                # Record metrics
                if _tool_call_counter:
                    _tool_call_counter.add(1, {"tool": tool_name})
                if _tool_duration_histogram:
                    _tool_duration_histogram.record(duration_ms, {"tool": tool_name})
                
                return result
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        """Async version of wrap_tool_call."""
        tool_name = request.tool_call.get("name", "unknown")
        tool_id = request.tool_call.get("id", "")
        
        with self.tracer.start_as_current_span(
            f"tool.{tool_name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("tool.call_id", tool_id)
            
            if request.tool:
                span.set_attribute("tool.description", (request.tool.description or "")[:200])
            
            if self.trace_content:
                args = request.tool_call.get("args", {})
                span.set_attribute("tool.input", str(args)[:1000])
            
            start_time = time.time()
            
            try:
                result = await handler(request)
                
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("tool.duration_ms", duration_ms)
                
                if self.trace_content and isinstance(result, ToolMessage):
                    span.set_attribute("tool.output", str(result.content)[:1000])
                
                # Check for error status
                if isinstance(result, ToolMessage):
                    status = getattr(result, "status", None)
                    if status == "error":
                        span.set_status(Status(StatusCode.ERROR, "Tool returned error"))
                    else:
                        span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status(Status(StatusCode.OK))
                
                # Record metrics
                if _tool_call_counter:
                    _tool_call_counter.add(1, {"tool": tool_name})
                if _tool_duration_histogram:
                    _tool_duration_histogram.record(duration_ms, {"tool": tool_name})
                
                return result
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    # ==================== Agent Invocation Helpers ====================
    
    def invoke_traced(
        self,
        agent,
        input_data: Dict[str, Any],
        agent_name: str = "agent",
        **attributes,
    ) -> Dict[str, Any]:
        """
        Invoke an agent with automatic tracing.
        
        This creates a root span that contains all nested operations including:
        - LLM calls from the main agent
        - Tool calls (which may invoke sub-agents)
        - LLM and tool calls from sub-agents
        
        Since a single middleware instance is shared across all agents,
        the context propagation ensures proper nesting of all spans.
        
        Args:
            agent: The agent to invoke
            input_data: Input data to pass to agent.invoke()
            agent_name: Name for the root span
            **attributes: Additional span attributes
        
        Returns:
            The result from agent.invoke()
        
        Example:
            middleware = OtelMiddleware(trace_content=True)
            
            # Create agents with shared middleware
            main_agent = create_agent(..., middleware=[middleware])
            sub_agent = create_agent(..., middleware=[middleware])
            
            # Invoke with tracing
            result = middleware.invoke_traced(
                main_agent,
                {"messages": [("user", "Hello")]},
                agent_name="main_orchestrator"
            )
        """
        with self.tracer.start_as_current_span(
            f"agent.{agent_name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("agent.name", agent_name)
            span.set_attribute("agent.type", "invocation")
            
            if self.service_name:
                span.set_attribute("service.name", self.service_name)
            
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
            
            # Track input message count
            messages = input_data.get("messages", [])
            if messages:
                span.set_attribute("agent.input_message_count", len(messages))
                if self.trace_content:
                    last_msg = messages[-1]
                    if isinstance(last_msg, tuple):
                        content = str(last_msg[1])[:500]
                    else:
                        content = str(getattr(last_msg, "content", str(last_msg)))[:500]
                    span.set_attribute("agent.input_message", content)
            
            start_time = time.time()
            
            try:
                result = agent.invoke(input_data)
                
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("agent.duration_ms", duration_ms)
                
                # Track output message count
                output_messages = result.get("messages", [])
                if output_messages:
                    span.set_attribute("agent.output_message_count", len(output_messages))
                
                span.set_status(Status(StatusCode.OK))
                return result
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    async def ainvoke_traced(
        self,
        agent,
        input_data: Dict[str, Any],
        agent_name: str = "agent",
        **attributes,
    ) -> Dict[str, Any]:
        """Async version of invoke_traced."""
        with self.tracer.start_as_current_span(
            f"agent.{agent_name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("agent.name", agent_name)
            span.set_attribute("agent.type", "invocation")
            
            if self.service_name:
                span.set_attribute("service.name", self.service_name)
            
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
            
            # Track input message count
            messages = input_data.get("messages", [])
            if messages:
                span.set_attribute("agent.input_message_count", len(messages))
                if self.trace_content:
                    last_msg = messages[-1]
                    if isinstance(last_msg, tuple):
                        content = str(last_msg[1])[:500]
                    else:
                        content = str(getattr(last_msg, "content", str(last_msg)))[:500]
                    span.set_attribute("agent.input_message", content)
            
            start_time = time.time()
            
            try:
                result = await agent.ainvoke(input_data)
                
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("agent.duration_ms", duration_ms)
                
                # Track output message count
                output_messages = result.get("messages", [])
                if output_messages:
                    span.set_attribute("agent.output_message_count", len(output_messages))
                
                span.set_status(Status(StatusCode.OK))
                return result
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

