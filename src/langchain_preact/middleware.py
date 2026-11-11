import copy
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command

PREDICTION_ARG = "prediction"

PREDICTION_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": (
        "The results this tool call might return. "
        "Fill it in last, after the real arguments."
    ),
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "observation": {
                "type": "string",
                "description": (
                    "A result that might come back from this call, "
                    "at a high level rather than the exact output."
                ),
            },
            "next_action": {
                "type": "string",
                "description": "What to do next if that result comes back.",
            },
        },
        "required": ["observation", "next_action"],
    },
}


def add_prediction_arg(tool: BaseTool) -> dict[str, Any]:
    schema = convert_to_openai_tool(tool)
    parameters = schema["function"].setdefault("parameters", {"type": "object"})
    properties = parameters.setdefault("properties", {})
    if PREDICTION_ARG in properties:
        msg = (
            f"Tool {tool.name!r} already has a {PREDICTION_ARG!r} argument, "
            "which PreActMiddleware would overwrite and strip before the tool runs."
        )
        raise ValueError(msg)
    properties[PREDICTION_ARG] = copy.deepcopy(PREDICTION_SCHEMA)
    parameters["required"] = [*parameters.get("required", []), PREDICTION_ARG]
    return schema


def _with_predictions(request: ModelRequest) -> ModelRequest:
    tools = [
        add_prediction_arg(t) if isinstance(t, BaseTool) else t for t in request.tools
    ]
    return request.override(tools=tools)


def _without_prediction(request: ToolCallRequest) -> ToolCallRequest:
    args = {k: v for k, v in request.tool_call["args"].items() if k != PREDICTION_ARG}
    return request.override(tool_call={**request.tool_call, "args": args})


class PreActMiddleware(AgentMiddleware):
    """Make the model predict each tool result through a required `prediction` arg.

    The arg is stripped before the tool runs but stays in the AIMessage tool calls,
    so the model sees its earlier predictions next to the real results. Provider
    built-in tools passed as dicts are left unchanged.
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(_with_predictions(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(_with_predictions(request))

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return handler(_without_prediction(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return await handler(_without_prediction(request))
