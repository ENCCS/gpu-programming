# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "GPU programming: why, when and how?"
copyright = "2023, The contributors"
author = "The contributors"
github_user = "ENCCS"
github_repo_name = "gpu-programming"  # auto-detected from dirname if blank
github_version = "main"
conf_py_path = "/content/"  # with leading and trailing slash

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # githubpages just adds a .nojekyll file
    "sphinx.ext.githubpages",
    "sphinx_lesson",
    # remove once sphinx_rtd_theme updated for contrast and accessibility:
    "sphinx_rtd_theme_ext_color_contrast",
    "sphinx.ext.todo",
]

# Settings for myst_nb:
# https://myst-nb.readthedocs.io/en/latest/use/execute.html#triggering-notebook-execution
# jupyter_execute_notebooks = "off"
# jupyter_execute_notebooks = "auto"   # *only* execute if at least one output is missing.
# jupyter_execute_notebooks = "force"
jupyter_execute_notebooks = "cache"

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "README*",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "jupyter_execute",
    "*venv*",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "img/ENCCS.jpg"
html_favicon = "img/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ["overrides.css"]

# HTML context:
from os.path import basename, dirname, realpath

html_context = {
    "display_github": True,
    "github_user": github_user,
    # Auto-detect directory name.  This can break, but
    # useful as a default.
    "github_repo": github_repo_name or basename(dirname(realpath(__file__))),
    "github_version": github_version,
    "conf_py_path": conf_py_path,
}

# Intersphinx mapping.  For example, with this you can use
# :py:mod:`multiprocessing` to link straight to the Python docs of that module.
# List all available references:
#   python -msphinx.ext.intersphinx https://docs.python.org/3/objects.inv
# extensions.append('sphinx.ext.intersphinx')
# intersphinx_mapping = {
#    #'python': ('https://docs.python.org/3', None),
#    #'sphinx': ('https://www.sphinx-doc.org/', None),
#    #'numpy': ('https://numpy.org/doc/stable/', None),
#    #'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
#    #'pandas': ('https://pandas.pydata.org/docs/', None),
#    #'matplotlib': ('https://matplotlib.org/', None),
#    'seaborn': ('https://seaborn.pydata.org/', None),
# }

# add few new directives
from sphinx_lesson.directives import _BaseCRDirective


class SignatureDirective(_BaseCRDirective):
    extra_classes = ["toggle-shown", "dropdown"]


class ParametersDirective(_BaseCRDirective):
    extra_classes = ["dropdown"]


class TypealongDirective(_BaseCRDirective):
    extra_classes = ["toggle-shown", "dropdown"]


DIRECTIVES = [SignatureDirective, ParametersDirective, TypealongDirective]


abbr_map = { }
abbr_map['grid'] = "In OpenCL and SYCL: NDRange."
abbr_map['block'] = "In OpenCL and SYCL: work-group."
abbr_map['warp'] = "In HIP: wavefront. In OpenCL and SYCL: sub-group."
abbr_map['thread'] = "In OpenCL and SYCL: work-item."
abbr_map['grid'] = "In OpenCL and SYCL: NDRange."
abbr_map['register'] = "In OpenCL and SYCL: private memory."
abbr_map['shared memory'] = "In OpenCL and SYCL: local memory (not to be confused with CUDA and HIP local memory)."
abbr_map['Grid'] = abbr_map['grid']
abbr_map['grids'] = abbr_map['grid']
abbr_map['Grids'] = abbr_map['grid']
abbr_map['Block'] = abbr_map['block']
abbr_map['blocks'] = abbr_map['block']
abbr_map['Blocks'] = abbr_map['block']
abbr_map['Warp'] = abbr_map['warp']
abbr_map['warps'] = abbr_map['warp']
abbr_map['Warps'] = abbr_map['warp']
abbr_map['Thread'] = abbr_map['thread']
abbr_map['threads'] = abbr_map['thread']
abbr_map['Threads'] = abbr_map['thread']


from docutils import nodes
from sphinx import roles
import logging
import sphinx.util.logging

class AutoAbbreviation(roles.Abbreviation):
    """A derivative of the Sphinx `abbr`, but with defaults.

    Used as :abbr:`name`.
    """

    _logger = sphinx.util.logging.getLogger('auto-abbr')

    def run(self):
        if '(' not in self.text:
            if self.text in abbr_map:
                options = self.options.copy()
                options['explanation'] = abbr_map[self.text]
                return [nodes.abbreviation(self.rawtext, self.text, **options)], []
            self._logger.warning("Abbreviation with no definition (%s): %s:%s",
                                 self.rawtext, *self.get_source_info())
        return super().run()


def setup(app):
    for obj in DIRECTIVES:
        app.add_directive(obj.cssname(), obj)

    app.add_role('abbr', AutoAbbreviation(), override=True)
    
import os
if os.environ.get('GITHUB_REF', '') == 'refs/heads/main':
    html_js_files = [
        ('https://plausible.io/js/script.js', {"data-domain": "enccs.github.io/gpu-programming", "defer": "defer"}),
    ]    
