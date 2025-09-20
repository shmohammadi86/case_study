"""Top-level src package for editable install.

Having this file allows imports like ::

    from src.study import ...

which are used by the upstream test-suite.  The actual study package is
located at ``src/study`` so we simply re-export it to make the dotted
reference work.
"""

import sys as _sys
from importlib import import_module as _imp

# Lazily proxy the study sub-package under the src namespace
# Import the sibling sub-package (src.study) so that callers can do
# ``from src.study import ...`` without explicitly adding the ``src``
# directory to their PYTHONPATH.
_study = _imp(__name__ + ".study")
_sys.modules[__name__ + ".study"] = _study

__all__ = ["study"]
