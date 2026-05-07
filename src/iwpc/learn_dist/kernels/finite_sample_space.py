from abc import ABC, abstractmethod
from itertools import product
from typing import Iterator, Any, Callable, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, ModuleList


class FiniteSampleSpace(ABC, Module):
    """
    Abstract class for representing the finite sample space for a probability distribution. Crucially every finite
    sample space can be mapped to the integers in [0, num_outcomes-1]. This mapping and its reverse are defined by
    outcome_to_idx and idx_to_outcome
    """
    def __init__(
        self,
        num_outcomes: int,
        dimension: int,
    ):
        """
        Parameters
        ----------
        num_outcomes
            The number of distinct elements in the sample space
        dimension
            The dimension of the samples in the space
        """
        super().__init__()
        self.num_outcomes = num_outcomes
        self.dimension = dimension

    @abstractmethod
    def outcome_to_idx(self, outcomes: Tensor) -> Tensor:
        """
        Parameters
        ----------
        samples
            A tensor of size (N, self.dimension)

        Returns
        -------
        Tensor
            An integer tensor of shape (N,) containing the indices for each sample in the sample space
        """

    @abstractmethod
    def idx_to_outcome(self, idxs) -> Tensor:
        """
        Mapping from an integer index to the set of possible outcomes

        Parameters
        ----------
        idxs
            An integer tensor of shape (N,) with entries in the range [0, K-1]

        Returns
        -------
        Tensor
            A tensor of shape (N, self.dimension)
        """

    def outcomes_iter(self) -> Iterator[Tensor]:
        """
        Returns
        -------
        Iterator[Tensor]
            Iterator over all possible outcomes in the sample space. Tensor is of shape (self.dimension,)
        """
        for idx in torch.arange(self.num_outcomes):
            yield self.idx_to_outcome(idx[None])[0]

    def __eq__(self, other: Any) -> bool:
        """
        Tests for equality with another FiniteSampleSpace based on the equality of the outcomes

        Parameters
        ----------
        other
            Any other object

        Returns
        -------
        bool
            Whether the other object is also a FiniteSampleSpace with the same set of outcomes
        """
        if isinstance(other, FiniteSampleSpace):
            if self.num_outcomes != other.num_outcomes:
                return False
            for o1, o2 in zip(self.outcomes_iter(), other.outcomes_iter()):
                if (o1 != o2).any():
                    return False
            return True
        return False

    def __hash__(self):
        return super().__hash__()

    def cut(self, cut_fn: Callable[[Tensor], bool]) -> "CutFiniteSampleSpace":
        """
        Parameters
        ----------
        cut_fn
            A function that accepts one of the elements of this sample space and returns whether the sample passes the
            cut

        Returns
        -------
        CutFiniteSampleSpace
            Containing the samples that passed the cut
        """
        return CutFiniteSampleSpace(self, cut_fn)

    def __and__(self, other: "FiniteSampleSpace") -> "ConcatenatedFiniteSampleSpace":
        """
        Synatactic sugar for the cartesian product of two FiniteSampleSpaces. Nested ConcatenatedFiniteSampleSpace
        instance are un-curried

        Parameters
        ----------
        other
            Another FiniteSampleSpace instance

        Returns
        -------
        ConcatenatedFiniteSampleSpace
            A sample space containing the cartesian product of two FiniteSampleSpace elements
        """
        if not isinstance(other, FiniteSampleSpace):
            raise ValueError(f"Cannot '&' objects of type FiniteSampleSpace and {type(other)}")

        sub_spaces = []
        for space in [self, other]:
            if isinstance(space, ConcatenatedFiniteSampleSpace):
                sub_spaces.extend(space.sub_spaces)
            else:
                sub_spaces.append(space)
        return ConcatenatedFiniteSampleSpace(sub_spaces)


class ConcatenatedFiniteSampleSpace(FiniteSampleSpace):
    """
    Sample space comprised of the cartesian product of a number of FiniteSampleSpace instances
    """
    def __init__(self, sub_spaces: Iterable[FiniteSampleSpace]):
        """
        Parameters
        ----------
        sub_spaces
            A list of FiniteSampleSpace instances
        """
        super().__init__(
            np.prod([s.num_outcomes for s in sub_spaces]),
            np.sum([s.dimension for s in sub_spaces]),
        )
        self.sub_spaces = ModuleList(sub_spaces)
        self.cum_num_outcomes = np.cumsum([0] + [s.dimension for s in sub_spaces])
        self.cond_edges = [slice(self.cum_num_outcomes[i], self.cum_num_outcomes[i + 1]) for i in range(len(sub_spaces))]
        self.register_buffer(
            'reversed_cumprod_num_variable_outcomes',
            torch.tensor(list(np.cumprod([s.num_outcomes for s in self.sub_spaces[::-1]])[::-1]) + [1])[1:]
        )

    def outcome_to_idx(self, outcomes) -> Tensor:
        """
        Parameters
        ----------
        samples
            A tensor of size (N, self.dimension)

        Returns
        -------
        Tensor
            An integer tensor of shape (N,) containing the indices for each sample in the sample space using the
            outcome_to_idx functions of the sub spaces.
        """
        sub_idxs = [s.outcome_to_idx(outcomes[:, slc]) for s, slc in zip(self.sub_spaces, self.cond_edges)]
        return sum(multiplier * idx for idx, multiplier in zip(sub_idxs, self.reversed_cumprod_num_variable_outcomes))

    def idx_to_outcome(self, idxs) -> Tensor:
        """
        Mapping from an integer index to the set of possible outcomes

        Parameters
        ----------
        idxs
            An integer tensor of shape (N,) with entries in the range [0, K-1]

        Returns
        -------
        Tensor
            A tensor of shape (N, self.dimension) using the idx_to_outcome of the sub-spaces
        """
        sub_idxs = torch.unravel_index(idxs, [int(s.num_outcomes) for s in self.sub_spaces])
        sub_samples = [s.idx_to_outcome(sub_idx) for s, sub_idx in zip(self.sub_spaces, sub_idxs)]
        return torch.concat(sub_samples, dim=1)

    def outcomes_iter(self) -> Iterator[Tensor]:
        """
        Returns
        -------
        Iterator[Tensor]
            Iterator over the cartesian product of all possible outcomes in the sub spaces
        """
        yield from map(torch.concat, product(*(s.outcomes_iter() for s in self.sub_spaces)))


class CutFiniteSampleSpace(FiniteSampleSpace):
    """
    Sample space comprised of the elements of a 'base_space' that pass some cut
    """
    def __init__(self, base_space: FiniteSampleSpace, cut_fn: Callable[[Tensor], bool]):
        """
        Parameters
        ----------
        base_space
            A FiniteSampleSpace instance
        cut_fn
            A function that accepts one of the elements of the base_space and returns whether the sample passes the
            cut
        """
        cut_idx = 0
        base_idx_to_cut_idx_map = []
        cut_idx_to_base_idx_map = []
        for base_idx, outcome in enumerate(base_space.outcomes_iter()):
            if cut_fn(outcome):
                base_idx_to_cut_idx_map.append(cut_idx)
                cut_idx_to_base_idx_map.append(base_idx)
                cut_idx += 1
            else:
                base_idx_to_cut_idx_map.append(-1)

        super().__init__(cut_idx, base_space.dimension)
        self.base_space = base_space
        self.cut_fn = cut_fn
        self.register_buffer(
            'base_idx_to_cut_idx_map',
            torch.tensor(base_idx_to_cut_idx_map, dtype=torch.int),
        )
        self.register_buffer(
            'cut_idx_to_base_idx_map',
            torch.tensor(cut_idx_to_base_idx_map, dtype=torch.int),
        )

    def outcome_to_idx(self, outcomes: Tensor) -> Tensor:
        """
        Parameters
        ----------
        samples
            A tensor of size (N, self.dimension)

        Returns
        -------
        Tensor
            An integer tensor of shape (N,) containing the indices for each sample in the sample space using the
            outcome_to_idx function of the base_space.
        """
        return self.base_idx_to_cut_idx_map[self.base_space.outcome_to_idx(outcomes)]

    def idx_to_outcome(self, idxs: Tensor) -> Tensor:
        """
        Mapping from an integer index to the set of possible outcomes

        Parameters
        ----------
        idxs
            An integer tensor of shape (N,) with entries in the range [0, K-1]

        Returns
        -------
        Tensor
            A tensor of shape (N, self.dimension) using the idx_to_outcome of the base space.
        """
        return self.base_space.idx_to_outcome(self.cut_idx_to_base_idx_map[idxs])


class ExplicitFiniteSampleSpace(FiniteSampleSpace):
    """
    FiniteSampleSpace over an explicit list of outcomes
    """
    def __init__(
        self,
        outcomes: Tensor | list[Tensor],
        outcome_to_idx_fn: Callable[[Tensor], Tensor],
    ):
        """
        Parameters
        ----------
        outcomes
            Either a tensor of shape (num_outcomes, self.dimension) or a list of num_outcomes tensors of shape (self.dimension,)
        outcome_to_idx_fn
            A callable that accepts a Tensor of shape (N, self.dimension) containing outcomes, and returns their
            respective indices within the outcomes tensor
        """
        outcomes = torch.stack(list(outcomes), dim=0).float()
        super().__init__(len(outcomes), outcomes.shape[1])
        self.register_buffer("outcomes", outcomes)
        self.outcome_to_idx_fn = outcome_to_idx_fn

    def idx_to_outcome(self, idxs: Tensor) -> Tensor:
        """
        Mapping from an integer index to the set of possible outcomes

        Parameters
        ----------
        idxs
            An integer tensor of shape (N,) with entries in the range [0, num_outcomes-1]

        Returns
        -------
        Tensor
            A tensor of shape (N, self.dimension)
        """
        return self.outcomes[idxs.int()]

    def outcome_to_idx(self, outcomes) -> Tensor:
        """
        Parameters
        ----------
        samples
            A tensor of size (N, self.dimension)

        Returns
        -------
        Tensor
            An integer tensor of shape (N,) containing the indices for each sample in the sample space
        """
        return self.outcome_to_idx_fn(outcomes)
