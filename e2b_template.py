from e2b_code_interpreter import Template

apt_dependencies = ["git", "curl"]

pip_requirements = [
    "requests",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "scipy",
    "markdown",
    "pillow",
]


def prepare_template():
    """
    This will build a template. It's just like image building, which is time consuming.
    make sure do this ahead of time.
    """
    template = (
        Template()
        .from_base_image()
        .apt_install(apt_dependencies)
        .pip_install(pip_requirements)
    )
    template = Template.build(template, alias="runner", cpu_count=1, memory_mb=128)
    return template


if __name__ == "__main__":
    template = prepare_template()
    print(template)
