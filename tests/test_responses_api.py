"""Tests for the Responses API adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any

import httpx
import pytest
from inline_snapshot import snapshot

from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.responses_api import SSE_CONTENT_TYPE

from .conftest import IsInstance, IsSameStr


async def simple_stream(messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[str]:
    """A simple stream function that emits two text chunks."""

    yield 'success '
    yield '(no tool calls)'


def parse_sse(body: str) -> list[dict[str, Any]]:
    """Parse Server-Sent Event payloads into structured data."""

    events: list[dict[str, Any]] = []
    for block in body.split('\n\n'):
        if not block.strip():
            continue
        event_type: str | None = None
        data: dict[str, Any] | None = None
        for line in block.split('\n'):
            if line.startswith('event: '):
                event_type = line.removeprefix('event: ')
            elif line.startswith('data: '):
                data = json.loads(line.removeprefix('data: '))
        if data is not None:
            events.append({'event': event_type, 'data': data})
    return events


@pytest.mark.anyio
async def test_to_responses_api() -> None:
    """The agent.to_responses_api() helper should expose a streaming endpoint."""

    agent = Agent(model=FunctionModel(stream_function=simple_stream))
    app = agent.to_responses_api()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
        payload = {'input': [{'type': 'input_text', 'text': 'Hello'}]}
        async with client.stream(
            'POST',
            '/responses',
            content=json.dumps(payload),
            headers={'Content-Type': 'application/json', 'Accept': SSE_CONTENT_TYPE},
        ) as response:
            assert response.status_code == HTTPStatus.OK
            body = await response.aread()

    events = parse_sse(body.decode())

    assert events == snapshot(
        [
            {
                'event': 'response.created',
                'data': {
                    'type': 'response.created',
                    'response': {
                        'id': (response_id := IsSameStr()),
                        'object': 'response',
                        'created_at': IsInstance(int),
                        'status': 'in_progress',
                        'model': None,
                        'output': [],
                        'parallel_tool_calls': True,
                        'instructions': '',
                        'metadata': {},
                        'reasoning': {'effort': None, 'generate_summary': None},
                        'tool_choice': 'auto',
                        'text': {'format': {'type': 'text'}},
                        'usage': None,
                    },
                    'sequence_number': 1,
                },
            },
            {
                'event': 'response.in_progress',
                'data': {
                    'type': 'response.in_progress',
                    'response': {
                        'id': response_id,
                        'object': 'response',
                        'created_at': IsInstance(int),
                        'status': 'in_progress',
                        'model': None,
                        'output': [],
                        'parallel_tool_calls': True,
                        'instructions': '',
                        'metadata': {},
                        'reasoning': {'effort': None, 'generate_summary': None},
                        'tool_choice': 'auto',
                        'text': {'format': {'type': 'text'}},
                        'usage': None,
                    },
                    'sequence_number': 2,
                },
            },
            {
                'event': 'response.output_item.added',
                'data': {
                    'type': 'response.output_item.added',
                    'output_index': 0,
                    'item': {
                        'type': 'message',
                        'id': (message_id := IsSameStr()),
                        'status': 'in_progress',
                        'role': 'assistant',
                        'content': [],
                    },
                    'sequence_number': 3,
                },
            },
            {
                'event': 'response.content_part.added',
                'data': {
                    'type': 'response.content_part.added',
                    'item_id': message_id,
                    'output_index': 0,
                    'content_index': 0,
                    'part': {'type': 'output_text', 'text': '', 'annotations': []},
                    'sequence_number': 4,
                },
            },
            {
                'event': 'response.output_text.delta',
                'data': {
                    'type': 'response.output_text.delta',
                    'item_id': message_id,
                    'output_index': 0,
                    'content_index': 0,
                    'delta': 'success ',
                    'sequence_number': 5,
                },
            },
            {
                'event': 'response.output_text.delta',
                'data': {
                    'type': 'response.output_text.delta',
                    'item_id': message_id,
                    'output_index': 0,
                    'content_index': 0,
                    'delta': '(no tool calls)',
                    'sequence_number': 6,
                },
            },
            {
                'event': 'response.output_text.done',
                'data': {
                    'type': 'response.output_text.done',
                    'item_id': message_id,
                    'output_index': 0,
                    'content_index': 0,
                    'text': 'success (no tool calls)',
                    'logprobs': [],
                    'sequence_number': 7,
                },
            },
            {
                'event': 'response.content_part.done',
                'data': {
                    'type': 'response.content_part.done',
                    'item_id': message_id,
                    'output_index': 0,
                    'content_index': 0,
                    'part': {'type': 'output_text', 'text': 'success (no tool calls)', 'annotations': []},
                    'sequence_number': 8,
                },
            },
            {
                'event': 'response.output_item.done',
                'data': {
                    'type': 'response.output_item.done',
                    'output_index': 0,
                    'item': {
                        'type': 'message',
                        'id': message_id,
                        'status': 'completed',
                        'role': 'assistant',
                        'content': [
                            {'type': 'output_text', 'text': 'success (no tool calls)', 'annotations': []},
                        ],
                    },
                    'sequence_number': 9,
                },
            },
            {
                'event': 'response.completed',
                'data': {
                    'type': 'response.completed',
                    'response': {
                        'id': response_id,
                        'object': 'response',
                        'created_at': IsInstance(int),
                        'status': 'completed',
                        'model': None,
                        'output': [
                            {
                                'type': 'message',
                                'id': message_id,
                                'status': 'completed',
                                'role': 'assistant',
                                'content': [
                                    {'type': 'output_text', 'text': 'success (no tool calls)', 'annotations': []},
                                ],
                            }
                        ],
                        'parallel_tool_calls': True,
                        'instructions': '',
                        'metadata': {},
                        'reasoning': {'effort': None, 'generate_summary': None},
                        'tool_choice': 'auto',
                        'text': {'format': {'type': 'text'}},
                        'usage': {
                            'input_tokens': IsInstance(int),
                            'output_tokens': IsInstance(int),
                            'total_tokens': IsInstance(int),
                            'output_tokens_details': {'reasoning_tokens': IsInstance(int)},
                        },
                    },
                    'sequence_number': 10,
                },
            },
        ]
    )
