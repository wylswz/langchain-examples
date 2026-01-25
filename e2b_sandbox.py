from e2b_code_interpreter import AsyncSandbox


async def create_sandbox():
    sandbox = await AsyncSandbox.create(template="runner")
    print(await sandbox.get_info())
    return sandbox


if __name__ == "__main__":
    import asyncio

    asyncio.run(create_sandbox())
