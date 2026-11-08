"""LongCat text-form ``<longcat_tool_call>`` blocks become real tool calls only for offered tools."""

import json
from types import SimpleNamespace

import pytest

from agent.transports import get_transport

TOOLS = {"terminal", "read_file"}


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # registers the transport
    return get_transport("chat_completions")


def _response(content, tool_calls=None, finish_reason="stop"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)], usage=None)


TWO_BLOCKS = (
    "Checking.\n"
    "<longcat_tool_call>terminal\n"
    "<longcat_arg_key>command</longcat_arg_key>\n"
    "<longcat_arg_value>cd /tmp && ls -la</longcat_arg_value>\n"
    "</longcat_tool_call>\n"
    "<longcat_tool_call>read_file\n"
    "<longcat_arg_key>path</longcat_arg_key>\n"
    "<longcat_arg_value>/tmp/a.txt</longcat_arg_value>\n"
    "</longcat_tool_call>"
)


def test_blocks_only_reply_becomes_tool_calls(transport):
    nr = transport.normalize_response(_response(TWO_BLOCKS), valid_tool_names=TOOLS)

    assert [tc.name for tc in nr.tool_calls] == ["terminal", "read_file"]
    assert json.loads(nr.tool_calls[0].arguments) == {"command": "cd /tmp && ls -la"}
    assert json.loads(nr.tool_calls[1].arguments) == {"path": "/tmp/a.txt"}
    assert nr.finish_reason == "tool_calls"
    assert nr.content == "Checking."


def test_unknown_tool_is_dropped_not_executed(transport):
    nr = transport.normalize_response(_response(TWO_BLOCKS), valid_tool_names={"read_file"})

    assert [tc.name for tc in nr.tool_calls] == ["read_file"]
    assert nr.content == "Checking."


def test_no_offered_tools_means_strip_only(transport):
    nr = transport.normalize_response(_response(TWO_BLOCKS))

    assert not nr.tool_calls
    assert nr.content == "Checking."
    assert nr.finish_reason == "stop"


def test_block_without_closing_tag_is_stripped_not_parsed(transport):
    text = "Hi\n<longcat_tool_call>terminal\n<longcat_arg_key>command</longcat_arg_key>\n<longcat_arg_value>ls"
    nr = transport.normalize_response(_response(text), valid_tool_names=TOOLS)

    assert nr.content == "Hi"
    assert not nr.tool_calls


def test_native_tool_calls_leave_content_alone(transport):
    native = SimpleNamespace(
        id="call_1", type="function", function=SimpleNamespace(name="terminal", arguments="{}"),
    )
    nr = transport.normalize_response(
        _response(TWO_BLOCKS, tool_calls=[native], finish_reason="tool_calls"), valid_tool_names=TOOLS,
    )

    assert [tc.id for tc in nr.tool_calls] == ["call_1"]
    assert nr.content == TWO_BLOCKS


def test_plain_text_is_untouched(transport):
    nr = transport.normalize_response(_response("just an answer"), valid_tool_names=TOOLS)

    assert not nr.tool_calls
    assert nr.content == "just an answer"
    assert nr.finish_reason == "stop"
