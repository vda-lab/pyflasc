import os
import matplotlib.pyplot

project = "FLASC"
author = "D.M. Bot and J. Aerts"
copyright = "2023, " + author
html_show_copyright = True
nitpicky = True
master_doc = "index"
linkcheck_ignore = [
    r"http://localhost:\d+/",
]

# Select nbsphinx and, if needed, other Sphinx extensions:
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",  # links to other Sphinx projects (e.g. NumPy)
    "sphinxcontrib.napoleon",
    "sphinx_last_updated_by_git",  # get "last updated" from Git
]

# These projects are also used for the sphinx_codeautolink extension:
intersphinx_mapping = {
    "IPython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3/", None),
    "hdbscan": ("https://hdbscan.readthedocs.io/en/latest/", None),
}

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ""

# List of arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
]

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None) %}
.. raw:: html
    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/spatialaudio/nbsphinx/blob/{{ env.config.release|e }}/{{ docname|e }}">{{ docname|e }}</a>.
      Interactive online version:
      <span style="white-space: nowrap;"><a href="https://mybinder.org/v2/gh/spatialaudio/nbsphinx/{{ env.config.release|e }}?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.</span>
      <script>
        if (document.location.host) {
          $(document.currentScript).replaceWith(
            '<a class="reference external" ' +
            'href="https://nbviewer.jupyter.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb">View in <em>nbviewer</em></a>.'
          );
        }
      </script>
    </div>
.. raw:: latex
    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""

# This is processed by Jinja2 and inserted after each notebook
nbsphinx_epilog = r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=None) %}
.. raw:: latex
    \nbsphinxstopnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{\dotfill\ \sphinxcode{\sphinxupquote{\strut
    {{ docname | escape_latex }}}} ends here.}}
"""

mathjax3_config = {
    "tex": {"tags": "ams", "useLabelIds": True},
}


# -- The settings below this line are not specific to nbsphinx ------------

try:
    from subprocess import check_output

    release = check_output(["git", "describe", "--tags", "--always"])
    release = release.decode().strip()
    today = check_output(["git", "show", "-s", "--format=%ad", "--date=short"])
    today = today.decode().strip()
except Exception:
    release = "<unknown>"
    today = "<unknown date>"

html_title = " version " + release

# -- Set HTML theme ---------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 5,
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for LaTeX output ---------------------------------------------

# See https://www.sphinx-doc.org/en/master/latex.html
latex_engine = "xelatex"
latex_elements = {
    "papersize": "a4paper",
    "printindex": "",
    "sphinxsetup": r"""
        %verbatimwithframe=false,
        %verbatimwrapslines=false,
        %verbatimhintsturnover=false,
        noteBorderColor={HTML}{E0E0E0},
        noteborder=1.5pt,
        warningBorderColor={HTML}{E0E0E0},
        warningborder=1.5pt,
        warningBgColor={HTML}{FBFBFB},
    """,
    "preamble": r"""
\usepackage[sc,osf]{mathpazo}
\linespread{1.05}  % see http://www.tug.dk/FontCatalogue/urwpalladio/
\renewcommand{\sfdefault}{pplj}  % Palatino instead of sans serif
\IfFileExists{zlmtt.sty}{
    \usepackage[light,scaled=1.05]{zlmtt}  % light typewriter font from lmodern
}{
    \renewcommand{\ttdefault}{lmtt}  % typewriter font from lmodern
}
""",
}
latex_table_style = ["booktabs"]
latex_documents = [
    (master_doc, "nbsphinx.tex", project, author, "howto"),
]
# latex_show_urls = 'footnote'
latex_show_pagerefs = True
