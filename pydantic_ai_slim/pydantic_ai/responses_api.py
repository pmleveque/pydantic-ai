"""Starlette integration that exposes an agent using the OpenAI Responses API protocol."""
from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Generic

from pydantic import BaseModel, ValidationError

from . import _utils
from .agent import AbstractAgent
from .messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from .models import KnownModelName, Model
from .output import OutputDataT, OutputSpec
from .settings import ModelSettings
from .tools import AgentDepsT, DeferredToolRequests
from .toolsets import AbstractToolset
from .usage import RunUsage, UsageLimits

try:  # pragma: no cover - dependency is optional
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse
    from starlette.routing import BaseRoute
    from starlette.types import ExceptionHandler, Lifespan
except ImportError as e:  # pragma: no cover - handled at runtime
    raise ImportError(
        'Please install the `starlette` package to use `Agent.to_responses_api()` method. '
        'You can use the `responses-api` optional group — `pip install "pydantic-ai-slim[responses-api]"`'
    ) from e

__all__ = [
    'ResponsesAPIApp',
    'handle_responses_api_request',
]


SSE_CONTENT_TYPE = 'text/event-stream'


class ResponsesAPIApp(Generic[AgentDepsT, OutputDataT], Starlette):
    """Starlette application that exposes an agent through the OpenAI Responses API contract."""

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        # Agent.iter parameters
        output_type: OutputSpec[Any] | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        # Starlette parameters
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any]] | None = None,
        on_shutdown: Sequence[Callable[[], Any]] | None = None,
        lifespan: Lifespan[ResponsesAPIApp[AgentDepsT, OutputDataT]] | None = None,
    ) -> None:
        super().__init__(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
        )

        async def endpoint(request: Request) -> Response:
            return await handle_responses_api_request(
                agent,
                request,
                output_type=output_type,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            )

        self.router.add_route('/responses', endpoint, methods=['POST'], name='create_response')


class _ResponsesRequest(BaseModel):
    model: str | None = None
    input: list[Any] | None = None
    messages: list[Any] | None = None
    instructions: str | None = None
    metadata: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


async def handle_responses_api_request(
    agent: AbstractAgent[AgentDepsT, Any],
    request: Request,
    *,
    output_type: OutputSpec[Any] | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    on_complete: Callable[[RunUsage], Any] | None = None,
) -> Response:
    """Process an OpenAI Responses API request and stream the agent output as SSE."""
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:  # pragma: no cover
        return Response(
            content=json.dumps({'detail': str(exc)}),
            media_type='application/json',
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        request_model = _ResponsesRequest.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover
        return Response(
            content=exc.json(),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    message_history = _messages_from_responses_request(request_model)

    stream = _responses_stream(
        agent,
        message_history=message_history,
        output_type=output_type,
        model=model,
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
        request_model=request_model,
        on_complete=on_complete,
    )

    return StreamingResponse(stream, media_type=SSE_CONTENT_TYPE)


async def _responses_stream(
    agent: AbstractAgent[AgentDepsT, Any],
    *,
    message_history: list[ModelMessage],
    output_type: OutputSpec[Any] | None,
    model: Model | KnownModelName | str | None,
    deps: AgentDepsT,
    model_settings: ModelSettings | None,
    usage_limits: UsageLimits | None,
    usage: RunUsage | None,
    infer_name: bool,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None,
    request_model: _ResponsesRequest,
    on_complete: Callable[[RunUsage], Any] | None,
) -> AsyncIterator[str]:
    response_state = _ResponseStreamState(
        model_name=request_model.model,
        parallel_tool_calls=request_model.parallel_tool_calls is not False,
        instructions=request_model.instructions,
        metadata=request_model.metadata,
        reasoning_request=request_model.reasoning,
    )

    yield response_state.emit_created()
    yield response_state.emit_in_progress()

    async with agent.iter(
        user_prompt=None,
        output_type=[output_type or agent.output_type, DeferredToolRequests],
        message_history=message_history,
        model=model,
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
    ) as run:
        async for node in run:
            if hasattr(node, 'stream'):
                async with node.stream(run.ctx) as stream:
                    async for agent_event in stream:
                        if isinstance(agent_event, FunctionToolResultEvent):
                            for sse_event in response_state.handle_tool_result(agent_event):
                                yield sse_event
                        else:
                            for sse_event in response_state.handle_agent_event(agent_event):
                                yield sse_event

        for part_index in list(response_state.active_parts):
            for event in response_state._finalize_part(part_index):
                yield event

        usage_summary = run.usage()
        response_state.set_usage(usage_summary)
        yield response_state.emit_completed()

        if on_complete is not None:
            if _utils.is_async_callable(on_complete):
                await on_complete(usage_summary)
            else:
                await asyncio.get_running_loop().run_in_executor(None, on_complete, usage_summary)


@dataclass
class _OutputTextState:
    item_id: str
    output_index: int
    content_index: int
    text: list[str] = field(default_factory=list)


@dataclass
class _ToolCallState:
    item_id: str
    call_id: str
    name: str
    output_index: int
    args: list[str] = field(default_factory=list)


@dataclass
class _ReasoningState:
    item_id: str
    output_index: int
    summary_index: int = 0
    summary_text: list[str] = field(default_factory=list)
    signature: str | None = None


@dataclass
class _ResponseStreamState:
    model_name: str | None
    parallel_tool_calls: bool
    instructions: str | None
    metadata: dict[str, Any] | None
    reasoning_request: dict[str, Any] | None

    response_id: str = field(default_factory=lambda: f'resp_{uuid.uuid4().hex}')
    created_at: int = field(default_factory=lambda: int(time.time()))
    sequence_number: int = 0
    output_index: int = 0
    active_parts: dict[int, Any] = field(default_factory=dict)
    completed_items: list[dict[str, Any]] = field(default_factory=list)
    usage: RunUsage | None = None
    def emit_created(self) -> str:
        return self._sse('response.created', {
            'type': 'response.created',
            'response': self._response_payload(status='in_progress', output=[]),
        })

    def emit_in_progress(self) -> str:
        return self._sse('response.in_progress', {
            'type': 'response.in_progress',
            'response': self._response_payload(status='in_progress', output=[]),
        })

    def emit_completed(self) -> str:
        return self._sse('response.completed', {
            'type': 'response.completed',
            'response': self._response_payload(status='completed', output=self.completed_items),
        })

    def set_usage(self, usage: RunUsage) -> None:
        self.usage = usage

    def handle_agent_event(self, agent_event: Any) -> list[str]:
        if isinstance(agent_event, PartStartEvent):
            return list(self._handle_part_start(agent_event))
        elif isinstance(agent_event, PartDeltaEvent):
            return list(self._handle_part_delta(agent_event))
        else:
            return []

    def _handle_part_start(self, event: PartStartEvent) -> Sequence[str]:
        index = event.index
        if index in self.active_parts:
            finish_events = list(self._finalize_part(index))
        else:
            finish_events = []

        part = event.part
        events: list[str] = []
        if isinstance(part, TextPart):
            state = _OutputTextState(
                item_id=part.id or f'msg_{uuid.uuid4().hex}',
                output_index=self.output_index,
                content_index=0,
            )
            self.output_index += 1
            self.active_parts[index] = state
            events.append(self._sse('response.output_item.added', {
                'type': 'response.output_item.added',
                'output_index': state.output_index,
                'item': {
                    'type': 'message',
                    'id': state.item_id,
                    'status': 'in_progress',
                    'role': 'assistant',
                    'content': [],
                },
            }))
            events.append(self._sse('response.content_part.added', {
                'type': 'response.content_part.added',
                'item_id': state.item_id,
                'output_index': state.output_index,
                'content_index': state.content_index,
                'part': {'type': 'output_text', 'text': '', 'annotations': []},
            }))
            if part.content:
                events.extend(self._handle_part_delta(PartDeltaEvent(index=index, delta=TextPartDelta(content_delta=part.content))))
        elif isinstance(part, ToolCallPart | BuiltinToolCallPart):
            call_id, item_id = _split_tool_call_identifier(part.tool_call_id)
            state = _ToolCallState(
                item_id=item_id,
                call_id=call_id,
                name=part.tool_name,
                output_index=self.output_index,
            )
            self.output_index += 1
            self.active_parts[index] = state
            events.append(self._sse('response.output_item.added', {
                'type': 'response.output_item.added',
                'output_index': state.output_index,
                'item': {
                    'type': 'function_call',
                    'id': state.item_id,
                    'call_id': state.call_id,
                    'name': state.name,
                    'arguments': '',
                    'status': 'in_progress',
                },
            }))
            if part.args:
                delta = part.args if isinstance(part.args, str) else json.dumps(part.args)
                events.append(self._sse('response.function_call_arguments.delta', {
                    'type': 'response.function_call_arguments.delta',
                    'item_id': state.item_id,
                    'output_index': state.output_index,
                    'delta': delta,
                }))
                state.args.append(delta)
        elif isinstance(part, ThinkingPart):
            state = _ReasoningState(
                item_id=part.id or f'rs_{uuid.uuid4().hex}',
                output_index=self.output_index,
            )
            self.output_index += 1
            if part.signature:
                state.signature = part.signature
            self.active_parts[index] = state
            events.append(self._sse('response.output_item.added', {
                'type': 'response.output_item.added',
                'output_index': state.output_index,
                'item': {
                    'type': 'reasoning',
                    'id': state.item_id,
                    'status': 'in_progress',
                    'summary': [],
                    'encrypted_content': state.signature,
                },
            }))
            events.append(self._sse('response.reasoning_summary_part.added', {
                'type': 'response.reasoning_summary_part.added',
                'item_id': state.item_id,
                'output_index': state.output_index,
                'summary_index': state.summary_index,
                'part': {'type': 'summary_text', 'text': ''},
            }))
            if part.content:
                events.extend(self._handle_part_delta(PartDeltaEvent(index=index, delta=ThinkingPartDelta(content_delta=part.content))))
        elif isinstance(part, BuiltinToolReturnPart):
            events.extend(self._emit_tool_return(part))
        return [*finish_events, *events]

    def _handle_part_delta(self, event: PartDeltaEvent) -> Sequence[str]:
        state = self.active_parts.get(event.index)
        if state is None:
            return []

        delta = event.delta
        events: list[str] = []
        if isinstance(delta, TextPartDelta) and isinstance(state, _OutputTextState):
            if delta.content_delta:
                state.text.append(delta.content_delta)
                events.append(self._sse('response.output_text.delta', {
                    'type': 'response.output_text.delta',
                    'item_id': state.item_id,
                    'output_index': state.output_index,
                    'content_index': state.content_index,
                    'delta': delta.content_delta,
                }))
        elif isinstance(delta, ToolCallPartDelta) and isinstance(state, _ToolCallState):
            if delta.tool_call_id:
                call_id, item_id = _split_tool_call_identifier(delta.tool_call_id)
                state.call_id = call_id
                state.item_id = item_id
            if delta.tool_name_delta:
                state.name = (state.name or '') + delta.tool_name_delta
            if delta.args_delta:
                args_delta = delta.args_delta if isinstance(delta.args_delta, str) else json.dumps(delta.args_delta)
                state.args.append(args_delta)
                events.append(self._sse('response.function_call_arguments.delta', {
                    'type': 'response.function_call_arguments.delta',
                    'item_id': state.item_id,
                    'output_index': state.output_index,
                    'delta': args_delta,
                }))
        elif isinstance(delta, ThinkingPartDelta) and isinstance(state, _ReasoningState):
            if delta.signature_delta:
                state.signature = delta.signature_delta
            if delta.content_delta:
                state.summary_text.append(delta.content_delta)
                events.append(self._sse('response.reasoning_summary_text.delta', {
                    'type': 'response.reasoning_summary_text.delta',
                    'item_id': state.item_id,
                    'output_index': state.output_index,
                    'summary_index': state.summary_index,
                    'delta': delta.content_delta,
                }))
        return events

    def _finalize_part(self, index: int) -> Sequence[str]:
        state = self.active_parts.pop(index, None)
        if state is None:
            return []

        events: list[str] = []
        if isinstance(state, _OutputTextState):
            text = ''.join(state.text)
            events.append(self._sse('response.output_text.done', {
                'type': 'response.output_text.done',
                'item_id': state.item_id,
                'output_index': state.output_index,
                'content_index': state.content_index,
                'text': text,
                'logprobs': [],
            }))
            events.append(self._sse('response.content_part.done', {
                'type': 'response.content_part.done',
                'item_id': state.item_id,
                'output_index': state.output_index,
                'content_index': state.content_index,
                'part': {'type': 'output_text', 'text': text, 'annotations': []},
            }))
            message_item = {
                'type': 'message',
                'id': state.item_id,
                'status': 'completed',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': text, 'annotations': []}],
            }
            events.append(self._sse('response.output_item.done', {
                'type': 'response.output_item.done',
                'output_index': state.output_index,
                'item': message_item,
            }))
            self.completed_items.append(message_item)
        elif isinstance(state, _ToolCallState):
            arguments = ''.join(state.args)
            events.append(self._sse('response.function_call_arguments.done', {
                'type': 'response.function_call_arguments.done',
                'item_id': state.item_id,
                'output_index': state.output_index,
                'arguments': arguments,
            }))
            tool_item = {
                'type': 'function_call',
                'id': state.item_id,
                'call_id': state.call_id,
                'name': state.name,
                'arguments': arguments,
                'status': 'completed',
            }
            events.append(self._sse('response.output_item.done', {
                'type': 'response.output_item.done',
                'output_index': state.output_index,
                'item': tool_item,
            }))
            self.completed_items.append(tool_item)
        elif isinstance(state, _ReasoningState):
            text = ''.join(state.summary_text)
            events.append(self._sse('response.reasoning_summary_text.done', {
                'type': 'response.reasoning_summary_text.done',
                'item_id': state.item_id,
                'output_index': state.output_index,
                'summary_index': state.summary_index,
                'text': text,
            }))
            events.append(self._sse('response.reasoning_summary_part.done', {
                'type': 'response.reasoning_summary_part.done',
                'item_id': state.item_id,
                'output_index': state.output_index,
                'summary_index': state.summary_index,
                'part': {'type': 'summary_text', 'text': text},
            }))
            reasoning_item = {
                'type': 'reasoning',
                'id': state.item_id,
                'status': 'completed',
                'encrypted_content': state.signature,
                'summary': [{'type': 'summary_text', 'text': text}],
            }
            events.append(self._sse('response.output_item.done', {
                'type': 'response.output_item.done',
                'output_index': state.output_index,
                'item': reasoning_item,
            }))
            self.completed_items.append(reasoning_item)
        return events

    def _response_payload(self, *, status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            'id': self.response_id,
            'object': 'response',
            'created_at': self.created_at,
            'status': status,
            'model': self.model_name,
            'output': output,
            'parallel_tool_calls': self.parallel_tool_calls,
            'instructions': self.instructions or '',
            'metadata': self.metadata or {},
            'reasoning': self._reasoning_summary(),
            'tool_choice': 'auto',
            'text': {'format': {'type': 'text'}},
            'usage': self._usage_dict(),
        }
        return payload

    def _usage_dict(self) -> dict[str, Any] | None:
        if self.usage is None:
            return None
        details = dict(self.usage.details)
        reasoning_tokens = details.pop('reasoning_tokens', 0)
        result = {
            'input_tokens': self.usage.input_tokens,
            'output_tokens': self.usage.output_tokens,
            'total_tokens': self.usage.input_tokens + self.usage.output_tokens,
            'output_tokens_details': {'reasoning_tokens': reasoning_tokens},
        }
        if self.usage.cache_read_tokens:
            result.setdefault('input_tokens_details', {})['cached_tokens'] = self.usage.cache_read_tokens
        if details:
            result['details'] = details
        return result

    def _reasoning_summary(self) -> dict[str, Any]:
        return {'effort': None, 'generate_summary': None}

    def _sse(self, event: str, data: dict[str, Any]) -> str:
        payload = json.dumps(data, ensure_ascii=False)
        self.sequence_number += 1
        if 'sequence_number' not in data:
            data = {**data, 'sequence_number': self.sequence_number}
            payload = json.dumps(data, ensure_ascii=False)
        return f'event: {event}\ndata: {payload}\n\n'

    def handle_tool_result(self, event: FunctionToolResultEvent) -> list[str]:
        result = event.result
        if not isinstance(result, ToolReturnPart):
            return []
        return self._emit_tool_return(result)

    def _emit_tool_return(self, result: ToolReturnPart | BuiltinToolReturnPart) -> list[str]:
        item_id = f'tool_{uuid.uuid4().hex}'
        output_index = self.output_index
        self.output_index += 1
        text = result.model_response_str()
        events = [
            self._sse('response.output_item.added', {
                'type': 'response.output_item.added',
                'output_index': output_index,
                'item': {
                    'type': 'message',
                    'id': item_id,
                    'status': 'in_progress',
                    'role': 'tool',
                    'tool_call_id': result.tool_call_id,
                    'content': [],
                },
            }),
            self._sse('response.content_part.added', {
                'type': 'response.content_part.added',
                'item_id': item_id,
                'output_index': output_index,
                'content_index': 0,
                'part': {'type': 'output_text', 'text': '', 'annotations': []},
            }),
            self._sse('response.output_text.delta', {
                'type': 'response.output_text.delta',
                'item_id': item_id,
                'output_index': output_index,
                'content_index': 0,
                'delta': text,
            }),
            self._sse('response.output_text.done', {
                'type': 'response.output_text.done',
                'item_id': item_id,
                'output_index': output_index,
                'content_index': 0,
                'text': text,
                'logprobs': [],
            }),
            self._sse('response.content_part.done', {
                'type': 'response.content_part.done',
                'item_id': item_id,
                'output_index': output_index,
                'content_index': 0,
                'part': {'type': 'output_text', 'text': text, 'annotations': []},
            }),
        ]
        tool_item = {
            'type': 'message',
            'id': item_id,
            'status': 'completed',
            'role': 'tool',
            'tool_call_id': result.tool_call_id,
            'content': [{'type': 'output_text', 'text': text, 'annotations': []}],
        }
        events.append(self._sse('response.output_item.done', {
            'type': 'response.output_item.done',
            'output_index': output_index,
            'item': tool_item,
        }))
        self.completed_items.append(tool_item)
        return events


def _split_tool_call_identifier(tool_call_id: str) -> tuple[str, str]:
    if '|' in tool_call_id:
        call_id, item_id = tool_call_id.split('|', 1)
        return call_id, item_id
    return tool_call_id, tool_call_id


def _messages_from_responses_request(request: _ResponsesRequest) -> list[ModelMessage]:
    raw_messages = request.messages if request.messages is not None else request.input or []
    result: list[ModelMessage] = []
    request_parts: list[ModelRequestPart] | None = None
    response_parts: list[ModelResponsePart] | None = None
    tool_names: dict[str, str] = {}
    pending_instructions = request.instructions

    for item in raw_messages:
        if not isinstance(item, Mapping):
            continue
        if 'role' in item:
            role = item['role']
            content = item.get('content')
            if role in {'user', 'system', 'developer'}:
                if request_parts is None:
                    request_parts = []
                    result.append(ModelRequest(parts=request_parts, instructions=pending_instructions))
                    pending_instructions = None
                    response_parts = None
                if role == 'system' or role == 'developer':
                    request_parts.append(SystemPromptPart(content=_extract_text_content(content)))
                else:
                    request_parts.append(UserPromptPart(content=_convert_user_content(content)))
            elif role == 'assistant':
                if response_parts is None:
                    response_parts = []
                    result.append(ModelResponse(parts=response_parts))
                    request_parts = None
                response_parts.extend(_convert_assistant_content(content))
            elif role == 'tool':
                if request_parts is None:
                    request_parts = []
                    result.append(ModelRequest(parts=request_parts, instructions=pending_instructions))
                    pending_instructions = None
                    response_parts = None
                call_id = item.get('tool_call_id') or item.get('call_id')
                tool_name = tool_names.get(call_id, item.get('name', 'tool'))
                request_parts.append(
                    ToolReturnPart(
                        tool_name=tool_name,
                        content=item.get('content'),
                        tool_call_id=call_id or f'call_{uuid.uuid4().hex}',
                    )
                )
        elif (t := item.get('type')) == 'function_call':
            if response_parts is None:
                response_parts = []
                result.append(ModelResponse(parts=response_parts))
                request_parts = None
            call_id = item.get('call_id') or f'call_{uuid.uuid4().hex}'
            args = item.get('arguments') or {}
            tool_name = item.get('name', 'tool')
            tool_names[call_id] = tool_name
            response_parts.append(
                ToolCallPart(
                    tool_name=tool_name,
                    args=args,
                    tool_call_id=item.get('id') or call_id,
                )
            )
        elif t == 'function_call_output':
            if request_parts is None:
                request_parts = []
                result.append(ModelRequest(parts=request_parts, instructions=pending_instructions))
                pending_instructions = None
                response_parts = None
            call_id = item.get('call_id')
            tool_name = tool_names.get(call_id, item.get('name', 'tool'))
            request_parts.append(
                ToolReturnPart(
                    tool_name=tool_name,
                    content=item.get('output'),
                    tool_call_id=call_id or f'call_{uuid.uuid4().hex}',
                )
            )

    return result


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return ''.join(
            block.get('text', '')
            for block in content
            if isinstance(block, Mapping) and block.get('type') in {'input_text', 'output_text'}
        )
    return ''


def _convert_user_content(content: Any) -> str | list[Any]:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence):
        return ''
    result: list[Any] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        block_type = block.get('type')
        if block_type in {'input_text', 'text'}:
            result.append(block.get('text', ''))
        elif block_type == 'input_image':
            url_field = block.get('image_url')
            url: str | None
            vendor_metadata: dict[str, Any] = {k: v for k, v in block.items() if k not in {'type', 'image_url', 'b64'}}
            if isinstance(url_field, Mapping):
                url = url_field.get('url')
                vendor_metadata.setdefault('detail', url_field.get('detail'))
            else:
                url = url_field
            if url:
                result.append(ImageUrl(url=url, vendor_metadata=vendor_metadata or None))
            elif (b64 := block.get('b64')):
                data = base64.b64decode(b64)
                media_type = block.get('media_type', 'image/png')
                result.append(BinaryContent(data=data, media_type=media_type))
        elif block_type == 'input_file':
            if (url := block.get('file_url')):
                result.append(DocumentUrl(url=url))
            elif (data_uri := block.get('file_data')) and isinstance(data_uri, str) and data_uri.startswith('data:'):
                header, _, payload = data_uri.partition(',')
                media_type = header.removeprefix('data:').removesuffix(';base64') or 'application/octet-stream'
                result.append(BinaryContent(data=base64.b64decode(payload), media_type=media_type))
    if not result:
        return ''
    if len(result) == 1 and isinstance(result[0], str):
        return result[0]
    return result


def _convert_assistant_content(content: Any) -> list[ModelResponsePart]:
    parts: list[ModelResponsePart] = []
    if isinstance(content, str):
        if content:
            parts.append(TextPart(content=content))
        return parts
    if not isinstance(content, Sequence):
        return parts
    for block in content:
        if not isinstance(block, Mapping):
            continue
        block_type = block.get('type')
        if block_type in {'output_text', 'text'}:
            parts.append(TextPart(content=block.get('text', '')))
    return parts

