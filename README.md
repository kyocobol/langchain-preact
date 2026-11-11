# LangChain PreAct

LangChain middleware that makes an agent predict the result of each tool call before it makes the call. Based on [PreAct: Prediction Enhances Agent's Planning Ability](https://arxiv.org/abs/2402.11534).

## Install

```bash
pip install langchain-preact
```

## Usage

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_preact import PreActMiddleware


@tool
def search(query: str) -> str:
    """Search the web."""
    ...


agent = create_agent(
    "anthropic:claude-sonnet-5-5",
    tools=[search],
    middleware=[PreActMiddleware()],
)
```

## How it works

`PreActMiddleware` adds a required `prediction` argument to the end of every tool schema. The model fills it with the results it expects and what it would do after each one:

```json
{
  "query": "PreAct paper authors",
  "prediction": [
    {
      "observation": "A list of authors from BUPT",
      "next_action": "Answer with the names"
    },
    {
      "observation": "No relevant results",
      "next_action": "Search for the arXiv ID instead"
    }
  ]
}
```

The middleware removes `prediction` before the tool runs, so the tool gets only its own arguments. The prediction stays in the message history, so on later turns the model sees what it predicted next to what actually happened. 

## Limits

- Provider built-in tools passed as dicts, such as `{"type": "web_search"}`, are left unchanged.
- A tool that already has a `prediction` argument raises `ValueError` on the first model call.
- The model receives tools as OpenAI-format dicts. List `PreActMiddleware` after middleware that reads tools as `BaseTool` objects, such as `LLMToolSelectorMiddleware`.
