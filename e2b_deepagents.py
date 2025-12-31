import os
from e2b_code_interpreter import Sandbox, CommandResult, Template


E2B_API_KEY = os.getenv("E2B_API_KEY")
if not E2B_API_KEY:
    raise Exception("E2B_API_KEY is not set")


def handle_result(result: CommandResult):
    if result.exit_code != 0:
        raise Exception(result.stderr)
    print(result.stdout)


def prepare_sandbox():
    return Sandbox.create(template="runner")


if __name__ == "__main__":
    sandbox = None
    try:
        sandbox = prepare_sandbox()
        sandbox.commands.run("python --version")
        sandbox.commands.run("pip --version")
        sandbox.commands.run("apt list")
    finally:
        sandbox.kill() if sandbox else None
