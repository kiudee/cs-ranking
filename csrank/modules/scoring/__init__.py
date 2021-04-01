"""High level implementation of scoring modules.

These are pytorch modules that take a list of instances and score each object
within each instance, taking its context into account. Only the high-level
assembly is done in this module.

One can easily derive ranking and choice estimators from a scorer, see
the implemented pytorch estimators such as ``FATEDiscreteObjectChooser`` for
examples.
"""

from .fate import FATEScoring

__all__ = ["FATEScoring"]