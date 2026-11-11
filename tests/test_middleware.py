import asyncio
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool, tool
from pydantic import Field

from langchain_preact import PreActMiddleware

PREDICTION = [
    {"observation": "a few matching documents", "next_action": "read the top one"},
    {"observation": "no results", "next_action": "broaden the query"},
]
WEB_SEARCH = {"type": "web_search"}


class ScriptedModel(BaseChatModel):
    replies: list[AIMessage]
    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(tools=list(tools), **kwargs)

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        self.calls.append({"messages": list(messages), "tools": kwargs.get("tools")})
        reply = self.replies[len(self.calls) - 1]
        return ChatResult(generations=[ChatGeneration(message=reply)])


def make_tools(received: list[tuple[str, dict]]):
    @tool
    def search(query: str, limit: int) -> str:
        """Search the corpus."""
        received.append(("search", {"query": query, "limit": limit}))
        return "2 documents"

    def lookup(key: str) -> str:
        received.append(("lookup", {"key": key}))
        return f"value of {key}"

    # A JSON-schema tool passes args straight to the function, so an unexpected
    # `prediction` kwarg would raise TypeError.
    strict = StructuredTool.from_function(
        func=lookup,
        name="lookup",
        description="Look up a key.",
        args_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    )
    return [search, strict]


def scripted_replies() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"query": "preact", "limit": 3, "prediction": PREDICTION},
                    "id": "call_search",
                },
                {
                    "name": "lookup",
                    "args": {"key": "k1", "prediction": PREDICTION},
                    "id": "call_lookup",
                },
            ],
        ),
        AIMessage(content="done"),
    ]


def build(middleware: list, received: list, replies: list[AIMessage] | None = None):
    model = ScriptedModel(replies=replies or scripted_replies())
    agent = create_agent(
        model,
        tools=[*make_tools(received), WEB_SEARCH],
        system_prompt="You are a careful researcher.",
        middleware=middleware,
    )
    return agent, model


INPUT = {"messages": [HumanMessage("find preact")]}


def bound_tool(model: ScriptedModel, name: str) -> dict:
    tools = model.calls[0]["tools"]
    return next(
        t["function"] for t in tools if t.get("function", {}).get("name") == name
    )


def test_bound_schema_appends_required_prediction_last():
    agent, model = build([PreActMiddleware()], [])
    agent.invoke(INPUT)

    search = bound_tool(model, "search")["parameters"]
    assert list(search["properties"]) == ["query", "limit", "prediction"]
    assert search["required"] == ["query", "limit", "prediction"]
    assert search["properties"]["query"]["type"] == "string"
    assert search["properties"]["limit"]["type"] == "integer"

    prediction = search["properties"]["prediction"]
    assert prediction["type"] == "array"
    assert prediction["minItems"] == 1
    item = prediction["items"]
    assert item["type"] == "object"
    assert list(item["properties"]) == ["observation", "next_action"]
    assert item["required"] == ["observation", "next_action"]
    assert item["properties"]["observation"]["type"] == "string"
    assert item["properties"]["next_action"]["type"] == "string"

    lookup = bound_tool(model, "lookup")["parameters"]
    assert list(lookup["properties"]) == ["key", "prediction"]
    assert lookup["required"] == ["key", "prediction"]

    assert WEB_SEARCH in model.calls[0]["tools"]


def test_tools_receive_only_their_original_args():
    received: list = []
    agent, _ = build([PreActMiddleware()], received)
    result = agent.invoke(INPUT)

    assert sorted(received) == [
        ("lookup", {"key": "k1"}),
        ("search", {"query": "preact", "limit": 3}),
    ]
    tool_messages = {
        m.name: m for m in result["messages"] if isinstance(m, ToolMessage)
    }
    assert tool_messages["search"].content == "2 documents"
    assert tool_messages["lookup"].content == "value of k1"
    assert all(m.status == "success" for m in tool_messages.values())


def predictions_in(messages: list[BaseMessage]) -> list:
    return [
        tc["args"]["prediction"]
        for m in messages
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]


def test_prediction_stays_in_history():
    agent, model = build([PreActMiddleware()], [])
    result = agent.invoke(INPUT)

    assert predictions_in(result["messages"]) == [PREDICTION, PREDICTION]
    assert predictions_in(model.calls[1]["messages"]) == [PREDICTION, PREDICTION]


def test_no_prompt_change_and_no_extra_model_calls():
    agent, with_mw = build([PreActMiddleware()], [])
    agent.invoke(INPUT)
    baseline_agent, without_mw = build([], [], replies=[AIMessage(content="done")])
    baseline_agent.invoke(INPUT)

    assert with_mw.calls[0]["messages"] == without_mw.calls[0]["messages"]
    assert with_mw.calls[0]["messages"][0].content == "You are a careful researcher."
    assert len(with_mw.calls) == len(scripted_replies())


def test_async_strips_prediction():
    received: list = []
    agent, model = build([PreActMiddleware()], received)
    result = asyncio.run(agent.ainvoke(INPUT))

    assert sorted(received) == [
        ("lookup", {"key": "k1"}),
        ("search", {"query": "preact", "limit": 3}),
    ]
    assert bound_tool(model, "search")["parameters"]["required"][-1] == "prediction"
    assert result["messages"][-1].content == "done"


def test_tool_with_prediction_arg_is_rejected():
    @tool
    def forecast(city: str, prediction: str) -> str:
        """Forecast the weather."""
        return prediction

    agent = create_agent(
        ScriptedModel(replies=[AIMessage(content="done")]),
        tools=[forecast],
        middleware=[PreActMiddleware()],
    )
    with pytest.raises(ValueError, match="forecast"):
        agent.invoke(INPUT)
