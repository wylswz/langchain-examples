from e2b_code_interpreter import Template
from e2b_code_interpreter import AsyncSandbox

pip_requirements = []
npm_requirements = [
    "@playwright/test",
]


def prepare_template():
    """
    This will build a template. It's just like image building, which is time consuming.
    make sure do this ahead of time.
    """
    template = Template().from_node_image().npm_install(npm_requirements)
    template = Template.build(template, alias="runner", cpu_count=2, memory_mb=2048)
    return template


if __name__ == "__main__":
    template = prepare_template()
    print(template)
