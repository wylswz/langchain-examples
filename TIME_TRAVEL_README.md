# LangGraph Time Travel Runner

A CLI tool that allows you to travel back in time through your LangGraph execution history and resume from any checkpoint.

## Overview

When running LangGraph agents with checkpointers, every step creates a checkpoint. The Time Travel Runner lets you:

- 📜 View all checkpoints in your execution history
- ⏮️ Resume execution from ANY checkpoint (not just the latest)
- 🔄 Retry failed steps or skip problematic ones
- 🐛 Debug complex workflows by going back to specific states

## Files

- `checkpointer-runner-time-travel.py` - Main time travel runner with interactive CLI
- `checkpointer.py` - Sample graph with random failures for testing
- `demo-time-travel.py` - Demo showing time travel in action

## How to Use

### 1. Run the Time Travel Runner

```bash
uv run python checkpointer-runner-time-travel.py
```

### 2. Understand the Checkpoint Display

The tool shows all checkpoints with detailed information:

```
================================================================================
CHECKPOINT HISTORY
================================================================================

[0] Checkpoint ID: 1f0afe3d-d766-6b74-8003-3c116018124e
    Step: 3
    Source: loop
    Next nodes: ()
    Messages count: 3
    Failures: 0
    Last message: this is step 3...

[1] Checkpoint ID: 1f0afe3d-d766-6674-8002-168aaab8be44
    Step: 2
    Source: loop
    Next nodes: ('step_3',)
    Messages count: 2
    Failures: 0
    Last message: this is step 2...
```

**Key Information:**
- **Index** `[0]`, `[1]`, etc. - What you enter to select this checkpoint
- **Step** - Which step in the execution this checkpoint represents
- **Next nodes** - What will execute next if you resume from here
- **Messages count** - How many messages are in the state
- **Last message** - Preview of the most recent message

### 3. Select a Checkpoint

When prompted:

```
Enter checkpoint index to resume from (0-4), or 'q' to quit:
```

Enter a number to resume from that checkpoint. The graph will continue execution from that point forward.

### 4. Handle Failures

If execution fails after resuming, the tool will:
- Show the error
- Refresh the checkpoint list
- Let you try again from a different checkpoint

## How It Works

### LangGraph Resume Mechanism

When you call `stream(None, config)` with a specific checkpoint config:

1. LangGraph loads the checkpoint state
2. Restores all channel values from that checkpoint
3. Continues execution from the next pending nodes
4. **Does NOT re-execute already completed steps**

### Time Travel Implementation

```python
# Get all checkpoints
checkpoints = list(cg.get_state_history(config))

# Each checkpoint has a config that can be used to resume
selected_checkpoint = checkpoints[idx]

# Resume from that specific checkpoint
stream = cg.stream(None, selected_checkpoint.config)
```

The magic is in `selected_checkpoint.config` - it contains the exact checkpoint ID to resume from.

## Use Cases

### 1. Retry Failed Steps

If a step fails randomly (like in our demo), you can:
- Keep retrying from the same checkpoint
- Eventually succeed without re-running earlier steps

### 2. Skip Problematic Steps

If a step consistently fails:
- Find the checkpoint AFTER that step
- Resume from there to skip it entirely

### 3. Debug Workflows

To understand what happened at a specific point:
- Resume from an earlier checkpoint
- Watch it execute again with fresh eyes
- Compare different execution paths

### 4. Rollback and Try Again

Made a mistake in a human-in-the-loop interaction?
- Go back to before that decision
- Make a different choice
- See how it affects the outcome

## Example Session

```bash
$ uv run python checkpointer-runner-time-travel.py

Starting initial run...
================================================================================
{'step_1': {'messages': [...]}}

❌ Initial run failed at random step: Random failure

Attempt 2 - Resuming from latest checkpoint...
--------------------------------------------------------------------------------
❌ Attempt 2 failed: Random failure

================================================================================
ENTERING TIME TRAVEL MODE
================================================================================

[0] Checkpoint ID: abc123...
    Step: 1
    Next nodes: ('step_2',)
    Messages count: 1

[1] Checkpoint ID: def456...
    Step: 0
    Next nodes: ('step_1',)
    Messages count: 0

Enter checkpoint index to resume from (0-1), or 'q' to quit: 0

================================================================================
RESUMING FROM CHECKPOINT 0
================================================================================

{'step_2': {'messages': [...]}}
{'step_3': {'messages': [...]}}

================================================================================
EXECUTION COMPLETED
================================================================================
```

## API Reference

### Main Functions

#### `display_checkpoints(thread_id="thread-1")`
Shows all checkpoints for a given thread in a formatted table.

Returns: List of `StateSnapshot` objects

#### `time_travel_resume(thread_id="thread-1")`
Interactive mode for selecting and resuming from checkpoints.

Features:
- Display all checkpoints
- Prompt for user selection
- Resume execution
- Handle errors gracefully
- Allow multiple time travel attempts

### StateSnapshot Structure

```python
StateSnapshot(
    values: dict,          # Current state values
    next: tuple[str, ...], # Next nodes to execute
    config: RunnableConfig,# Config to resume from this point
    metadata: dict         # Step, source, writes, etc.
)
```

## Tips and Tricks

1. **Start with latest checkpoint** - Index [0] is always the most recent
2. **Check "Next nodes"** - Shows what will execute if you resume
3. **Watch message count** - Helps identify where in execution you are
4. **Use 'q' to quit** - Exit time travel mode safely anytime

## Limitations

- Only works with compiled graphs that have a checkpointer
- Requires the same thread_id to access checkpoint history  
- Cannot modify past checkpoints, only resume from them
- Some side effects (external API calls, etc.) may re-execute

## Learn More

- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Human-in-the-Loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)
- [State Management](https://langchain-ai.github.io/langgraph/concepts/state/)

