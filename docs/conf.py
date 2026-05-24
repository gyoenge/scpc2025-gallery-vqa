project = "SCPC2025 Gallery VQA"
author = "gyoenge"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

myst_enable_extensions = ["colon_fence", "deflist"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "SCPC2025 Gallery VQA"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
