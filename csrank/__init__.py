from importlib.metadata import version

# We should re-evaluate if we really want to re-export everything here and then
# use __all__ properly.

from .choicefunction import *  # noqa
from .core import *  # noqa
from .dataset_reader import *  # noqa
from .discretechoice import *  # noqa
from .objectranking import *  # noqa

__version__ = version(__name__)
