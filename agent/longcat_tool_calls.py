"""Parse LongCat's text-form tool-call blocks into real tool calls.

LongCat models served without native function calling (e.g. ``meituan/longcat-2.5:free`` via Nous)
may write calls into the message body instead of ``tool_calls``::

    <longcat_tool_call>terminal
    <longcat_arg_key>command</longcat_arg_key>
    <longcat_arg_value>ls -la</longcat_arg_value>
    </longcat_tool_call>

Only complete blocks naming a tool in ``valid_tool_names`` become calls; truncated, malformed or
unknown-tool blocks are dropped, never guessed. All blocks are removed from the visible text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection

MARKER = "<longcat_tool_call>"

# A block ends at its closing tag, at the next opening tag, or at end of text (truncated output).
_BLOCK_RE = re.compile(
    r"<longcat_tool_call>.*?(?:</longcat_tool_call>|(?=<longcat_tool_call>)|\Z)",
    re.DOTALL,
)
_COMPLETE_RE = re.compile(r"<longcat_tool_call>(.*?)</longcat_tool_call>", re.DOTALL)
_ARG_RE = re.compile(
    r"<longcat_arg_key>(.*?)</longcat_arg_key>\s*<longcat_arg_value>(.*?)</longcat_arg_value>",
    re.DOTALL,
)


def strip_longcat_tool_calls(text: str) -> str:
    """``text`` without any ``<longcat_tool_call>`` block; unchanged when none is present."""
    if MARKER not in text:
        return text
    return _BLOCK_RE.sub("", text).strip()


def parse_longcat_tool_calls(text: str, valid_tool_names: Collection[str]) -> list[tuple[str, str]]:
    """``(name, json_arguments)`` for each complete block that names a whitelisted tool."""
    if MARKER not in text:
        return []
    calls: list[tuple[str, str]] = []
    for body in _COMPLETE_RE.findall(text):
        head, _, rest = body.partition("<longcat_arg_key>")
        name = head.strip()
        if not name or name not in valid_tool_names:
            continue
        args = {k.strip(): v for k, v in _ARG_RE.findall("<longcat_arg_key>" + rest)} if rest else {}
        calls.append((name, json.dumps(args, ensure_ascii=False)))
    return calls
