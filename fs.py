# Filesystem utilities for sandboxed operations
# 
# This module provides path resolution and security utilities
# for operating within a sandboxed directory structure.

from pathlib import Path


def get_sandbox_root(config: dict, base_dir: Path) -> Path:
    """Get the sandbox root path for the current session.
    
    Args:
        config: LangGraph config dict containing thread_id in configurable
        base_dir: Base directory for all sandboxes
        
    Returns:
        Path to the session's sandbox directory (created if not exists)
    """
    session_id = config.get("configurable", {}).get("thread_id", "default")
    sandbox_path = base_dir / session_id
    sandbox_path.mkdir(parents=True, exist_ok=True)
    return sandbox_path


def resolve_path(sandbox_root: Path, absolute_path: str) -> Path:
    """Resolve an absolute path to the sandboxed filesystem.
    
    Maps paths like /main.py or /src/utils.py to actual sandbox locations.
    Prevents directory traversal attacks.
    
    Args:
        sandbox_root: The sandbox root directory
        absolute_path: User-provided absolute path (e.g., /main.py)
        
    Returns:
        Resolved path within the sandbox
        
    Raises:
        ValueError: If path would escape the sandbox
    """
    # Normalize: ensure path starts with /
    if not absolute_path.startswith("/"):
        absolute_path = "/" + absolute_path
    
    # Remove leading slash and clean up
    clean_path = absolute_path.lstrip("/")
    
    # Handle root directory
    if not clean_path:
        return sandbox_root
    
    # Resolve the full path
    full_path = (sandbox_root / clean_path).resolve()
    
    # Security: ensure path stays within sandbox
    try:
        full_path.relative_to(sandbox_root.resolve())
    except ValueError:
        raise ValueError(f"Access denied: path '{absolute_path}' is outside the filesystem")
    
    return full_path


def to_absolute_path(sandbox_root: Path, real_path: Path) -> str:
    """Convert a real filesystem path back to an absolute path.
    
    Args:
        sandbox_root: The sandbox root directory
        real_path: Actual filesystem path within the sandbox
        
    Returns:
        Absolute path string (e.g., /main.py)
    """
    rel_path = real_path.relative_to(sandbox_root)
    return "/" + str(rel_path)


def format_size(size: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5KB", "2.3MB")
    """
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size/1024:.1f}KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size/1024/1024:.1f}MB"
    else:
        return f"{size/1024/1024/1024:.1f}GB"


def add_line_numbers(content: str) -> str:
    """Add line numbers to file content.
    
    Args:
        content: File content
        
    Returns:
        Content with line numbers prefixed
    """
    lines = content.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i+1:>{width}}| {line}" for i, line in enumerate(lines))
