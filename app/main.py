from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import CodingAgent

app = FastAPI()


class AgentRunRequest(BaseModel):
    prompt: str


class AgentRunResponse(BaseModel):
    success: bool
    text: str


@app.post("/agent/run", response_model=AgentRunResponse)
async def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    """Run one coding task through the Claude Agent SDK and return the result."""
    agent = CodingAgent()
    result = await agent.run(request.prompt)
    return AgentRunResponse(success=result.success, text=result.text)
