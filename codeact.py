# CodeAct Agent - An agent that writes and executes code to solve problems
# It uses file system tools to read and write files in a sandboxed environment
# and uses Python runtime to execute code
#
# Design: The agent works with absolute paths (e.g., /main.py, /src/utils.py)
# but all operations are transparently sandboxed by session ID.

import re
import subprocess
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from python_runner import PythonRunner

# =============================================================================
# Configuration
# =============================================================================

SANDBOX_BASE_DIR = Path(__file__).parent / "sandbox"
MAX_OUTPUT_LENGTH = 10000  # Max characters for command output
PYTHON_TIMEOUT = 30  # seconds

# Default dependencies for the Python environment
# Can be overridden when creating the agent
DEFAULT_DEPENDENCIES: list[str] = []

# Cache for PythonRunner instances (keyed by session_id)
_runner_cache: dict[str, PythonRunner] = {}


# =============================================================================
# Sandbox Path Resolution (Internal - Hidden from Agent)
# =============================================================================

def _get_sandbox_root(config: dict) -> Path:
    """Get the sandbox root path for the current session.
    
    Internal function - the agent doesn't know about this.
    """
    session_id = config.get("configurable", {}).get("thread_id", "default")
    sandbox_path = SANDBOX_BASE_DIR / session_id
    sandbox_path.mkdir(parents=True, exist_ok=True)
    return sandbox_path


def _resolve_path(config: dict, path: str) -> Path:
    """Resolve an absolute path to the sandboxed filesystem.
    
    The agent uses absolute paths like /main.py or /src/utils.py.
    This function maps them to the actual sandbox location.
    
    Internal function - the agent doesn't know about this.
    """
    sandbox_root = _get_sandbox_root(config)
    
    # Normalize: ensure path starts with / and remove redundant slashes
    if not path.startswith("/"):
        path = "/" + path
    
    # Remove the leading slash and any path traversal attempts
    clean_path = path.lstrip("/")
    
    # Handle empty path (root directory)
    if not clean_path:
        return sandbox_root
    
    # Resolve the full path
    full_path = (sandbox_root / clean_path).resolve()
    
    # Security: ensure the resolved path is within the sandbox
    try:
        full_path.relative_to(sandbox_root.resolve())
    except ValueError:
        raise ValueError(f"Access denied: path '{path}' is outside the filesystem")
    
    return full_path


def _format_size(size: int) -> str:
    """Format file size in human-readable format."""
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size/1024:.1f}KB"
    else:
        return f"{size/1024/1024:.1f}MB"


def _get_runner(config: dict) -> PythonRunner:
    """Get or create a PythonRunner for the current session.
    
    Runners are cached by session_id for reuse.
    """
    session_id = config.get("configurable", {}).get("thread_id", "default")
    
    if session_id not in _runner_cache:
        sandbox_path = _get_sandbox_root(config)
        _runner_cache[session_id] = PythonRunner(
            sandbox_path=sandbox_path,
            dependencies=DEFAULT_DEPENDENCIES.copy(),
            timeout=PYTHON_TIMEOUT
        )
    
    return _runner_cache[session_id]


# =============================================================================
# File System Tools
# =============================================================================

@tool
def read_file(
    path: Annotated[str, "Absolute path to the file (e.g., /main.py, /src/utils.py)"],
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """Read the contents of a file.
    
    Args:
        path: Absolute path to the file
        
    Returns:
        The file contents with line numbers
    """
    try:
        full_path = _resolve_path(config, path)
        
        if not full_path.exists():
            return f"Error: File '{path}' does not exist"
        
        if not full_path.is_file():
            return f"Error: '{path}' is not a file"
        
        content = full_path.read_text(encoding="utf-8")
        
        # Add line numbers
        lines = content.split("\n")
        width = len(str(len(lines)))
        numbered_lines = [f"{i+1:>{width}}| {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)
        
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(
    path: Annotated[str, "Absolute path to the file (e.g., /main.py, /src/utils.py)"],
    content: Annotated[str, "Content to write to the file"],
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """Write content to a file. Creates parent directories if needed.
    
    Args:
        path: Absolute path to the file
        content: Content to write
        
    Returns:
        Success or error message
    """
    try:
        full_path = _resolve_path(config, path)
        
        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        full_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{path}'"
        
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def edit_file(
    path: Annotated[str, "Absolute path to the file"],
    old_string: Annotated[str, "The exact string to find and replace"],
    new_string: Annotated[str, "The string to replace with"],
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """Edit a file by replacing a specific string. The old_string must match exactly.
    
    Args:
        path: Absolute path to the file
        old_string: Exact string to find (must be unique in file)
        new_string: String to replace with
        
    Returns:
        Success or error message
    """
    try:
        full_path = _resolve_path(config, path)
        
        if not full_path.exists():
            return f"Error: File '{path}' does not exist"
        
        content = full_path.read_text(encoding="utf-8")
        
        # Check how many times the old_string appears
        count = content.count(old_string)
        
        if count == 0:
            return f"Error: String not found in file. Make sure you're using the exact string including whitespace."
        
        if count > 1:
            return f"Error: String appears {count} times. Provide more context to make it unique."
        
        # Perform the replacement
        new_content = content.replace(old_string, new_string, 1)
        full_path.write_text(new_content, encoding="utf-8")
        
        return f"Successfully edited '{path}'"
        
    except Exception as e:
        return f"Error editing file: {str(e)}"


@tool
def list_directory(
    path: Annotated[str, "Absolute path to directory (use '/' for root)"] = "/",
    *, config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """List contents of a directory.
    
    Args:
        path: Absolute path to directory, defaults to root '/'
        
    Returns:
        Directory listing with file types and sizes
    """
    try:
        full_path = _resolve_path(config, path)
        
        if not full_path.exists():
            return f"Error: Directory '{path}' does not exist"
        
        if not full_path.is_dir():
            return f"Error: '{path}' is not a directory"
        
        items = []
        for item in sorted(full_path.iterdir()):
            # Skip hidden files starting with underscore (internal temp files)
            if item.name.startswith("_"):
                continue
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            else:
                size_str = _format_size(item.stat().st_size)
                items.append(f"📄 {item.name} ({size_str})")
        
        if not items:
            return f"Directory '{path}' is empty"
        
        return f"Contents of '{path}':\n" + "\n".join(items)
        
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def create_directory(
    path: Annotated[str, "Absolute path to directory to create"],
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """Create a directory (and any necessary parent directories).
    
    Args:
        path: Absolute path to create
        
    Returns:
        Success or error message
    """
    try:
        full_path = _resolve_path(config, path)
        full_path.mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory '{path}'"
        
    except Exception as e:
        return f"Error creating directory: {str(e)}"


@tool  
def delete_file(
    path: Annotated[str, "Absolute path to the file to delete"],
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """Delete a file.
    
    Args:
        path: Absolute path to the file
        
    Returns:
        Success or error message
    """
    try:
        full_path = _resolve_path(config, path)
        
        if not full_path.exists():
            return f"Error: File '{path}' does not exist"
        
        if full_path.is_dir():
            return f"Error: '{path}' is a directory, not a file"
        
        full_path.unlink()
        return f"Successfully deleted '{path}'"
        
    except Exception as e:
        return f"Error deleting file: {str(e)}"


# =============================================================================
# Search Tools
# =============================================================================

@tool
def grep(
    pattern: Annotated[str, "Regex pattern to search for"],
    path: Annotated[str, "Absolute path to file or directory to search"] = "/",
    include: Annotated[str, "Glob pattern for files to include (e.g., '*.py')"] = "*",
    context: Annotated[int, "Number of context lines before and after match"] = 0,
    *, config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """Search for a pattern in files using regex.
    
    Args:
        pattern: Regular expression pattern to search for
        path: Absolute path to file or directory to search in
        include: Glob pattern to filter files (e.g., '*.py', '*.txt')
        context: Number of lines of context around matches
        
    Returns:
        Matching lines with file paths and line numbers
    """
    try:
        full_path = _resolve_path(config, path)
        sandbox_root = _get_sandbox_root(config)
        regex = re.compile(pattern)
        results = []
        
        def search_file(fp: Path):
            try:
                # Get the path relative to sandbox, then format as absolute
                rel_to_sandbox = fp.relative_to(sandbox_root)
                display_path = "/" + str(rel_to_sandbox)
                
                content = fp.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")
                
                for i, line in enumerate(lines):
                    if regex.search(line):
                        # Add context
                        start = max(0, i - context)
                        end = min(len(lines), i + context + 1)
                        
                        for j in range(start, end):
                            prefix = ">" if j == i else " "
                            results.append(f"{display_path}:{j+1}{prefix} {lines[j]}")
                        
                        if context > 0:
                            results.append("---")
                            
            except Exception:
                pass  # Skip files that can't be read
        
        if full_path.is_file():
            search_file(full_path)
        else:
            for fp in full_path.rglob(include):
                if fp.is_file() and not fp.name.startswith("_"):
                    search_file(fp)
        
        if not results:
            return f"No matches found for pattern '{pattern}'"
        
        output = "\n".join(results)
        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"
        
        return output
        
    except re.error as e:
        return f"Invalid regex pattern: {str(e)}"
    except Exception as e:
        return f"Error searching: {str(e)}"


@tool
def find_files(
    pattern: Annotated[str, "Glob pattern to match files (e.g., '*.py', 'test_*.py')"],
    path: Annotated[str, "Absolute path to directory to search in"] = "/",
    *, config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """Find files matching a glob pattern.
    
    Args:
        pattern: Glob pattern (e.g., '*.py', '**/*.txt', 'test_*')
        path: Absolute path to directory to search in
        
    Returns:
        List of matching file paths (absolute)
    """
    try:
        full_path = _resolve_path(config, path)
        sandbox_root = _get_sandbox_root(config)
        
        if not full_path.exists():
            return f"Error: Directory '{path}' does not exist"
        
        matches = []
        for match in full_path.rglob(pattern):
            # Skip internal temp files
            if match.name.startswith("_"):
                continue
            matches.append(match)
        
        if not matches:
            return f"No files found matching '{pattern}'"
        
        results = []
        for match in sorted(matches):
            # Convert to absolute path format
            rel_to_sandbox = match.relative_to(sandbox_root)
            abs_path = "/" + str(rel_to_sandbox)
            if match.is_dir():
                results.append(f"📁 {abs_path}/")
            else:
                results.append(f"📄 {abs_path}")
        
        return f"Found {len(results)} matches:\n" + "\n".join(results)
        
    except Exception as e:
        return f"Error finding files: {str(e)}"


# =============================================================================
# Code Execution Tools
# =============================================================================

@tool
def run_python(
    code: Annotated[str, "Python code to execute"],
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """Execute Python code in an isolated virtual environment.
    
    The code runs with '/' as the working directory.
    Pre-configured packages are available (see environment info).
    Files created/modified by the code will persist in the filesystem.
    
    Args:
        code: Python code to execute
        
    Returns:
        stdout and stderr from execution, or error message
    """
    runner = _get_runner(config)
    result = runner.run(code)
    return result.format_output(MAX_OUTPUT_LENGTH)


@tool
def run_shell(
    command: Annotated[str, "Shell command to execute"],
    config: Annotated[RunnableConfig, InjectedToolArg()]
) -> str:
    """Execute a shell command.
    
    The working directory is '/'. Some dangerous commands are blocked.
    
    Args:
        command: Shell command to run
        
    Returns:
        Command output or error message
    """
    sandbox = _get_sandbox_root(config)
    
    # Block dangerous commands
    dangerous_patterns = [
        r'\brm\s+-rf\s+/',
        r'\bsudo\b',
        r'\bchmod\b.*777',
        r'>\s*/dev/',
        r'\bdd\b',
        r'\bmkfs\b',
        r'\bformat\b',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Error: Command blocked for safety reasons"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT,
            cwd=str(sandbox)
        )
        
        output_parts = []
        
        if result.stdout:
            output_parts.append(result.stdout)
        
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")
        
        if result.returncode != 0 and not result.stderr:
            output_parts.append(f"Exit code: {result.returncode}")
        
        if not output_parts:
            output_parts.append("(No output)")
        
        output = "\n".join(output_parts)
        
        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {PYTHON_TIMEOUT} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


# =============================================================================
# Agent Definition
# =============================================================================

def _build_system_prompt(dependencies: list[str]) -> str:
    """Build system prompt with available packages info."""
    
    if dependencies:
        packages_section = f"""## Available Python Packages
The following packages are pre-installed and ready to import:
{', '.join(dependencies)}

You can use these packages directly in your code without installation."""
    else:
        packages_section = """## Python Environment
Standard library is available. No additional packages are pre-installed."""
    
    return f"""You are a CodeAct agent - an AI that solves problems by writing and executing code.

## Your Environment
You have access to:
- A filesystem where you can read, write, and execute code
- An isolated Python virtual environment

All file paths are absolute, starting from root '/'.

Examples:
- /main.py
- /src/utils.py  
- /data/input.csv
- /tests/test_main.py

{packages_section}

## Your Workflow
1. **Understand** the problem thoroughly
2. **Plan** your approach before coding
3. **Implement** step by step, testing as you go
4. **Verify** your solution works correctly
5. **Iterate** if needed based on results

## Best Practices
- Write clean, readable code with comments
- Test your code after writing it
- Handle errors gracefully
- Break complex problems into smaller steps
- Use the right tool for each task
- **All outputs from generated code MUST be printed to stdout using print()**
- **Do NOT read or write files in generated code - return results via stdout only**

## Available Tools

### File Operations
- `read_file(path)`: Read file contents with line numbers
- `write_file(path, content)`: Create or overwrite a file
- `edit_file(path, old_string, new_string)`: Make targeted edits
- `list_directory(path)`: List directory contents (default: '/')
- `create_directory(path)`: Create a directory
- `delete_file(path)`: Delete a file

### Search
- `grep(pattern, path, include, context)`: Search for regex pattern in files
- `find_files(pattern, path)`: Find files by glob pattern

### Execution
- `run_python(code)`: Execute Python code in the virtual environment
- `run_shell(command)`: Run shell commands

## Important
- All paths must be absolute (start with '/')
- You start with an empty filesystem
- Files you create persist throughout the session
- The first run_python call may take a moment while the environment initializes"""


# Collect all tools
tools = [
    read_file,
    write_file,
    edit_file,
    list_directory,
    create_directory,
    delete_file,
    grep,
    find_files,
    run_python,
    run_shell,
]


def create_agent(model_name: str = "gpt-4o", default_packages: list[str] = None):
    """Create the CodeAct agent graph.
    
    Args:
        model_name: OpenAI model to use
        default_packages: List of Python packages to pre-install in each session
    """
    global DEFAULT_DEPENDENCIES
    if default_packages:
        DEFAULT_DEPENDENCIES = default_packages
    
    # Build system prompt with package info
    system_prompt = _build_system_prompt(DEFAULT_DEPENDENCIES)
    
    # Initialize model with tools
    model = ChatOpenAI(model=model_name, temperature=0)
    model_with_tools = model.bind_tools(tools)
    
    # Define the agent node
    def agent_node(state: MessagesState):
        messages = state["messages"]
        
        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Define routing logic after agent response
    def should_continue(state: MessagesState) -> Literal["tools", "human_review"]:
        last_message = state["messages"][-1]
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "human_review"
    
    # Human-in-the-loop node
    def human_review_node(state: MessagesState) -> Command[Literal["agent", "__end__"]]:
        """Interrupt for human review. User can say 'yes' to end or provide more instructions."""
        user_response = interrupt(
            "Task completed. Reply 'yes' to confirm, or provide further instructions:"
        )
        
        # If user confirms, end the conversation
        if user_response.strip().lower() in ["yes", "y", "done", "ok", "confirmed"]:
            return Command(goto=END)
        
        # Otherwise, treat as new instructions and continue
        return Command(
            goto="agent",
            update={"messages": [HumanMessage(content=user_response)]}
        )
    
    # Build the graph
    graph = StateGraph(MessagesState)
    
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("human_review", human_review_node)
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    # human_review routes via Command in the node itself
    
    # Compile with checkpointer for conversation memory
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Interactive CLI for the CodeAct agent."""
    import uuid

    
    agent = create_agent(default_packages=["pandas", "numpy", "scikit-learn"])
    thread_id = str(uuid.uuid4())
    
    print(f"Session ID: {thread_id}")
    print(f"(Internal sandbox: {SANDBOX_BASE_DIR / thread_id})\n")
    
    while True:
        try:
            user_input = 'Predict next 10 numbers: [1, 4, 8, 3, 7, 2, 8, 10, 14, 7, 21, 16, 22, 26, 19, 27, 32], you might need to do some regression'
            print(user_input)
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("Goodbye! 👋")
                break
            
            if user_input.lower() == "new":
                thread_id = str(uuid.uuid4())
                # Clear the runner cache for clean session
                if thread_id in _runner_cache:
                    del _runner_cache[thread_id]
                print(f"\n🆕 New session: {thread_id}\n")
                continue
            
            config = {"configurable": {"thread_id": thread_id}}
            
            print("\nAgent: ", end="", flush=True)
            
            # Stream the response
            for event in agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values"
            ):
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage):
                    if last_message.content:
                        print(last_message.content)
                    if last_message.tool_calls:
                        for tc in last_message.tool_calls:
                            print(f"\n🔧 Using tool: {tc['name']}")
                elif isinstance(last_message, ToolMessage):
                    content = last_message.content
                    if len(content) > 500:
                        content = content[:500] + "..."
                    print(f"📋 Result: {content}")
            
            print()
            
            # Handle human-in-the-loop: check if agent is waiting for user input
            while True:
                state = agent.get_state(config)
                if not state.next:  # No next nodes means graph has ended
                    break
                
                # Get the interrupt value (prompt for user)
                if state.tasks and state.tasks[0].interrupts:
                    interrupt_value = state.tasks[0].interrupts[0].value
                    print(f"\n🔄 {interrupt_value}")
                    human_response = input("You: ").strip()
                    
                    if not human_response:
                        human_response = "yes"  # Default to confirming
                    
                    # Resume the graph with user's response
                    print("\nAgent: ", end="", flush=True)
                    for event in agent.stream(
                        Command(resume=human_response),
                        config=config,
                        stream_mode="values"
                    ):
                        last_message = event["messages"][-1]
                        if isinstance(last_message, AIMessage):
                            if last_message.content:
                                print(last_message.content)
                            if last_message.tool_calls:
                                for tc in last_message.tool_calls:
                                    print(f"\n🔧 Using tool: {tc['name']}")
                        elif isinstance(last_message, ToolMessage):
                            content = last_message.content
                            if len(content) > 500:
                                content = content[:500] + "..."
                            print(f"📋 Result: {content}")
                    print()
                else:
                    break
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
