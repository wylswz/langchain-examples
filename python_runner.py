# Python Runner with Virtual Environment Support
#
# Simple API: Create runner with dependencies, then just call run().
# The runner initializes once per session and caches state for performance.
#
# Usage:
#     runner = PythonRunner(sandbox_path, dependencies=["requests", "pandas"])
#     result = runner.run("import pandas; print(pandas.__version__)")
#     # First call initializes venv + installs deps (slow)
#     # Subsequent calls skip setup (fast)

import os
import sys
import subprocess
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of a Python code execution."""
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    timed_out: bool = False
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.return_code == 0 and not self.timed_out and not self.error
    
    def format_output(self, max_length: int = 10000) -> str:
        """Format the execution result for display."""
        if self.error:
            return f"Error: {self.error}"
        
        if self.timed_out:
            return "Error: Execution timed out"
        
        parts = []
        
        if self.stdout:
            parts.append(f"STDOUT:\n{self.stdout}")
        
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr}")
        
        if self.return_code != 0:
            parts.append(f"Exit code: {self.return_code}")
        
        if not parts:
            parts.append("(No output)")
        
        output = "\n\n".join(parts)
        
        if len(output) > max_length:
            output = output[:max_length] + "\n... (output truncated)"
        
        return output


@dataclass
class PythonRunner:
    """Isolated Python execution with virtual environment.
    
    Dependencies are declared in constructor. The runner initializes
    the venv on first run() call and caches state - subsequent calls
    skip initialization for performance.
    
    Usage:
        runner = PythonRunner(sandbox_path, dependencies=["requests"])
        result = runner.run("import requests; print('ok')")
    """
    
    sandbox_path: Path
    dependencies: list[str] = field(default_factory=list)
    timeout: int = 30
    
    # Internal state
    _venv_path: Path = field(init=False)
    _python_path: Path = field(init=False)
    _pip_path: Path = field(init=False)
    _deps_hash_file: Path = field(init=False)
    _initialized: bool = field(init=False, default=False)
    
    def __post_init__(self):
        self.sandbox_path = Path(self.sandbox_path)
        self._venv_path = self.sandbox_path / ".venv"
        
        # Platform-specific paths
        if sys.platform == "win32":
            self._python_path = self._venv_path / "Scripts" / "python.exe"
            self._pip_path = self._venv_path / "Scripts" / "pip.exe"
        else:
            self._python_path = self._venv_path / "bin" / "python"
            self._pip_path = self._venv_path / "bin" / "pip"
        
        self._deps_hash_file = self._venv_path / ".deps_hash"
    
    def _compute_deps_hash(self) -> str:
        """Compute hash of dependencies for cache invalidation."""
        deps_str = json.dumps(sorted(self.dependencies), sort_keys=True)
        return hashlib.sha256(deps_str.encode()).hexdigest()[:16]
    
    def _needs_setup(self) -> bool:
        """Check if venv needs initialization or deps need updating."""
        # Already initialized in this session
        if self._initialized:
            return False
        
        # No venv exists
        if not self._venv_path.exists() or not self._python_path.exists():
            return True
        
        # No dependencies - venv exists, we're good
        if not self.dependencies:
            return False
        
        # Check if deps hash matches (detect dependency changes)
        if not self._deps_hash_file.exists():
            return True
        
        stored_hash = self._deps_hash_file.read_text().strip()
        return stored_hash != self._compute_deps_hash()
    
    def _setup(self) -> tuple[bool, str]:
        """Initialize venv and install dependencies (called automatically)."""
        try:
            self.sandbox_path.mkdir(parents=True, exist_ok=True)
            
            # Skip if already set up
            if not self._needs_setup():
                self._initialized = True
                return True, "Environment ready"
            
            # Create venv
            if not self._venv_path.exists():
                result = subprocess.run(
                    [sys.executable, "-m", "venv", str(self._venv_path)],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode != 0:
                    return False, f"Failed to create venv: {result.stderr}"
            
            # Install dependencies
            if self.dependencies:
                # Upgrade pip quietly
                subprocess.run(
                    [str(self._pip_path), "install", "--upgrade", "pip", "-q"],
                    capture_output=True,
                    timeout=120
                )
                
                # Install deps
                result = subprocess.run(
                    [str(self._pip_path), "install", "-q"] + self.dependencies,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    return False, f"Failed to install dependencies: {result.stderr}"
                
                # Cache the deps hash
                self._deps_hash_file.write_text(self._compute_deps_hash())
            
            self._initialized = True
            return True, "Environment initialized"
            
        except subprocess.TimeoutExpired:
            return False, "Setup timed out"
        except Exception as e:
            return False, f"Setup failed: {e}"
    
    def run(self, code: str) -> ExecutionResult:
        """Execute Python code in the virtual environment.
        
        First call initializes the venv (may be slow).
        Subsequent calls are fast.
        
        Args:
            code: Python code to execute
            
        Returns:
            ExecutionResult with stdout, stderr, return_code
        """
        # Auto-initialize on first run
        success, message = self._setup()
        if not success:
            return ExecutionResult(error=message)
        
        script_path = self.sandbox_path / "_temp_script.py"
        
        try:
            script_path.write_text(code, encoding="utf-8")
            
            result = subprocess.run(
                [str(self._python_path), str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.sandbox_path),
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "VIRTUAL_ENV": str(self._venv_path),
                }
            )
            
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(timed_out=True)
        except Exception as e:
            return ExecutionResult(error=str(e))
        finally:
            if script_path.exists():
                script_path.unlink()
    
    @property
    def is_ready(self) -> bool:
        """Check if environment is initialized."""
        return self._initialized and self._python_path.exists()


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create runner with dependencies
        runner = PythonRunner(
            sandbox_path=Path(tmpdir),
            dependencies=["requests"]
        )
        
        # Just run code - initialization happens automatically
        print("First run (initializes venv)...")
        result = runner.run("""
import requests
print(f"requests version: {requests.__version__}")
""")
        print(result.format_output())
        
        # Second run is fast (cached)
        print("\nSecond run (cached, fast)...")
        result = runner.run("print('Hello from cached venv!')")
        print(result.format_output())
