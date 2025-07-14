from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template


def load_template(template_file: str) -> Template:
    template_path = Path(template_file)
    env = Environment(loader=FileSystemLoader(template_path.parent), autoescape=True)
    return env.get_template(template_path.name)
