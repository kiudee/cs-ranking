from abc import ABCMeta
from abc import abstractmethod


class Tunable(metaclass=ABCMeta):
    @abstractmethod
    def set_tunable_parameters(self, **point):
        """ Set tunable parameters of the algorithm to the values provided.

        Parameters
        ----------
        point : dict
            Dictionary containing parameter values

        """
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Tunable:
            has_set_tunable_param = any(
                "set_tunable_parameters" in B.__dict__ for B in C.__mro__
            )
            if has_set_tunable_param:
                return True
        return NotImplemented
