from checkpointer import graph, MyException
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime

# This is a demo for time travel in checkpointer
# It allows you to resume from any checkpoint in the execution history if the graph fails at any step

saver = MemorySaver()
cg = graph.compile(checkpointer=saver)

def run():
    """Initial run of the graph"""
    stream = cg.stream({"messages": []}, {"configurable": {"thread_id": "thread-1"}})
    for event in stream:
        print(event)

def resume():
    """Resume from the latest checkpoint"""
    stream = cg.stream(None, {"configurable": {"thread_id": "thread-1"}})
    for event in stream:
        print(event)

def display_checkpoints(thread_id="thread-1"):
    """Display all checkpoints for a given thread"""
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get all checkpoints
    checkpoints = list(cg.get_state_history(config))
    
    if not checkpoints:
        print("No checkpoints found!")
        return []
    
    print("\n" + "="*80)
    print("CHECKPOINT HISTORY")
    print("="*80)
    
    for idx, checkpoint in enumerate(checkpoints):
        metadata = checkpoint.metadata or {}
        checkpoint_id = checkpoint.config.get("configurable", {}).get("checkpoint_id", "N/A")
        
        # Extract useful info
        step = metadata.get("step", "N/A")
        source = metadata.get("source", "N/A")
        writes = metadata.get("writes", {})
        
        print(f"\n[{idx}] Checkpoint ID: {checkpoint_id}")
        print(f"    Step: {step}")
        print(f"    Source: {source}")
        print(f"    Next nodes: {checkpoint.next}")
        
        # Show messages in state
        messages = checkpoint.values.get("messages", [])
        failures = checkpoint.values.get("failures", 0)
        print(f"    Messages count: {len(messages)}")
        print(f"    Failures: {failures}")
        
        if messages:
            last_msg = messages[-1]
            content = getattr(last_msg, 'content', str(last_msg))
            print(f"    Last message: {content[:50]}...")
        
        if writes:
            print(f"    Writes: {writes}")
    
    print("\n" + "="*80)
    return checkpoints

def time_travel_resume(thread_id="thread-1"):
    """Allow user to select which checkpoint to resume from"""
    checkpoints = display_checkpoints(thread_id)
    
    if not checkpoints:
        return
    
    while True:
        try:
            user_input = input(f"\nEnter checkpoint index to resume from (0-{len(checkpoints)-1}), or 'q' to quit: ")
            
            if user_input.lower() == 'q':
                print("Exiting time travel mode.")
                return
            
            idx = int(user_input)
            
            if idx < 0 or idx >= len(checkpoints):
                print(f"Invalid index. Please enter a number between 0 and {len(checkpoints)-1}")
                continue
            
            # Get the selected checkpoint
            selected_checkpoint = checkpoints[idx]
            
            print(f"\n{'='*80}")
            print(f"RESUMING FROM CHECKPOINT {idx}")
            print(f"{'='*80}\n")
            
            # Resume from the selected checkpoint
            stream = cg.stream(None, selected_checkpoint.config)
            for event in stream:
                print(event)
            
            print(f"\n{'='*80}")
            print("EXECUTION COMPLETED")
            print(f"{'='*80}\n")
            
            # Ask if user wants to continue time traveling
            continue_input = input("Time travel again? (y/n): ")
            if continue_input.lower() != 'y':
                break
            
            # Refresh checkpoint list
            checkpoints = display_checkpoints(thread_id)
            
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
        except MyException as e:
            print(f"\n❌ Execution failed: {e}")
            print("The graph hit an error. You can try resuming from a different checkpoint.\n")
            
            # Refresh checkpoint list after failure
            checkpoints = display_checkpoints(thread_id)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            break


if __name__ == "__main__":
    print("Starting initial run...")
    print("="*80)
    
    # First run - will fail at step_2
    try:
        run()
    except MyException as e:
        print(f"\n❌ Initial run failed at random step: {e}\n")
    
    # Try resuming a few times to create more checkpoints
    for i in range(2, 5):
        try:
            print(f"\nAttempt {i} - Resuming from latest checkpoint...")
            print("-"*80)
            resume()
            print("\n✅ Execution completed successfully!")
            break
        except MyException as e:
            print(f"\n❌ Attempt {i} failed: {e}")
    
    # Now enter time travel mode
    print("\n\n" + "="*80)
    print("ENTERING TIME TRAVEL MODE")
    print("="*80)
    print("\nYou can now select any checkpoint to resume from.")
    print("This allows you to go back to any point in the execution history.\n")
    
    time_travel_resume()

