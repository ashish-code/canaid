"""External-system tools the agents can call.

Phase 4 ships the mock CRM and three lookup tools. Phase 5 adds a
"create handoff ticket" tool for the escalation agent. Phase 9 swaps
the in-memory CRM for DynamoDB-backed via the same tool surface.

The tools are LangChain ``@tool``-decorated functions because that gives
us, for free:
  * JSON-schema generation (used in Bedrock toolConfig)
  * pydantic input validation
  * ``on_tool_start`` / ``on_tool_end`` callbacks during ``astream_events``
    — the API server uses these to surface tool calls as SSE frames.
"""

from canaid.tools.crm import MockCRM, get_crm
from canaid.tools.lookup_tools import LOOKUP_TOOLS, TOOL_MAP

__all__ = ["LOOKUP_TOOLS", "TOOL_MAP", "MockCRM", "get_crm"]
