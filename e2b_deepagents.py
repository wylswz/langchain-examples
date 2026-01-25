import os
from typing import TypedDict
from attr import dataclass
from e2b import FileType
from e2b_code_interpreter import AsyncSandbox
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import AgentMiddleware
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from typing import Any
import uuid
import aiofiles.os as aos
import aiofiles

from pydantic import BaseModel

E2B_API_KEY = os.getenv("E2B_API_KEY")
if not E2B_API_KEY:
    raise Exception("E2B_API_KEY is not set")

# you will need to run
# e2b_sandbox.py to create one, or just create it on website
SANDBIX_ID = "irwuen86cxputdtbfbbin"

MODEL = ChatOpenAI(model="gpt-4.1")


ENV_PROMPT = """
You have access to a full sandbox environment with a bash and nodejs runtime, this allows you to execute
arbitrary code in the sandbox environment to accomplish the tasks.

The bash tool is extremely powerful, you can do anything with it

for example, you can run a command to install a package:
```bash
npm install playwright
```

or you can run a command to execute a script:
```bash
node scripts/generate_report.js
```

to write code, you can simple echo into a file, for example:
```bash
echo "print('Hello, world!')" > hello.py
```

or just replace few lines of code in a file, for example:
```bash
sed -i 's/print("Hello, world!")/print("Hello, world! 2")/' hello.py
```

besides, you can use the sandbox as a scratchpad or todo list.

so always try to leverage the bash tool, and perform tasks by writing codes.
"""

SYSTEM_PROMPT = f"""
You are a business analyst that can help with tasks related to the business analysis. When you are given a goal,
you will need to create a plan with several sub-tasks, and assign the sub-tasks to the worker agents.

worker agents will have access to the sandbox environment, and can execute arbitrary code to accomplish the tasks.
The worker agents will return the results of the tasks to you, and you will need to review the results and create a
final report.

If the results are not good enough, you will need to create a new plan with new sub-tasks, 
and assign the sub-tasks to the worker agents.

You will need to summarize the final report and return it to the user. The final report can be a markdown, or an html page, 
depending on the user's request.

{ENV_PROMPT}
"""

WORKER_PROMPT = f"""
You are a worker that can help with tasks related to the worker. The work includes
- research: do a deep research online to help with the business analysis.
- analysis: analyze the data to help with the business analysis.
- presentation: present the data to the end user. The presentation can be a report, a website, or a slide.

If you create any files, you should save the files to the sandbox environment, and return the file path when
you are done.

Before you finish, always try to download the files you created so that the users can access it.

{ENV_PROMPT}
"""


class BashResult(TypedDict):
    stdout: str
    stderr: str
    exit_code: int


@tool
async def bash(cmd: str, runtime: ToolRuntime) -> BashResult:
    """Run a bash command"""
    print(f"Running command: {cmd}")
    try:
        ret = await SandboxMiddleware.SANDBOX.commands.run(cmd)
        return {
            "stdout": ret.stdout,
            "stderr": ret.stderr,
            "exit_code": ret.exit_code,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": 1,
        }


async def walk_dir(sandbox: AsyncSandbox, path: str) -> list[str]:
    info = await sandbox.files.get_info(path)
    if info.type == FileType.FILE:
        return [path]
    ret = []
    entries = await sandbox.files.list(path)
    for entry in entries:
        if entry.type == FileType.FILE:
            ret.append(entry.path)
        elif entry.type == FileType.DIR:
            ret.extend(await walk_dir(sandbox, entry.path))
    return ret


@tool
async def download_artifact(path: str, tool_rt: ToolRuntime) -> str:
    """
    Download an artifact from the sandbox environment. It supports recursive download for path, so you don't have to compress if there
    are just a few files.
    """
    try:
        rid = tool_rt.config["configurable"]["thread_id"]
        output_dir = os.path.join("output", rid)
        await aos.makedirs(output_dir, exist_ok=True)
        entries = await walk_dir(SandboxMiddleware.SANDBOX, path)
        for entry in entries:
            content = await SandboxMiddleware.SANDBOX.files.read(entry, format="bytes")
            # Make entry path relative by stripping leading slash if present
            relative_entry = entry.lstrip("/")
            local_path = os.path.join(output_dir, relative_entry)
            parent_dir = os.path.dirname(local_path)
            if parent_dir != "":
                await aos.makedirs(parent_dir, exist_ok=True)
            async with aiofiles.open(local_path, "wb") as f:
                await f.write(content)
        return output_dir
    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"Error downloading artifact: {str(e)}"


class SandboxMiddleware(AgentMiddleware[AgentState]):
    SANDBOX: AsyncSandbox

    def __init__(self):
        pass

    async def abefore_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        SandboxMiddleware.SANDBOX = await AsyncSandbox.connect(sandbox_id=SANDBIX_ID)

    async def aafter_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return {}


graph = create_agent(
    MODEL,
    system_prompt=WORKER_PROMPT,
    middleware=[SandboxMiddleware()],
    tools=[
        bash,
        download_artifact,
        TavilySearch(max_results=10),
    ],
)


async def main():
    stream = graph.astream(
        {
            "messages": [
                HumanMessage(
                    content="perform a business analysis on the company: langgenius"
                )
            ],
        },
        stream_mode="messages",
    )
    async for message, _ in stream:
        print(message.content, end="")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
