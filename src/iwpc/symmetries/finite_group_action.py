from typing import Iterable, Tuple

from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.group_action_element import GroupActionElement, Identity


class FiniteGroupAction(GroupAction):
    """
    Generic implementation of a finite group action
    """
    def __init__(self, non_id_elements: Iterable[GroupActionElement]):
        """
        Parameters
        ----------
        non_id_elements
            An iterable of the non-identity GroupActionElements in the group action
        """
        super().__init__()
        self.elements = (Identity(), *non_id_elements)
        for i, element in enumerate(self.elements):
            self.register_module(f"non_id_element_{i}", element)

    def batch(self) -> Tuple[GroupActionElement, ...]:
        """
        Returns
        -------
        Tuple[GroupActionElement, ...]
            All the elements in the group action, including the identity element
        """
        return self.elements

    def __len__(self):
        """
        Returns
        -------
        int
            The number of elements in the group action including the identity element
        """
        return len(self.elements)
