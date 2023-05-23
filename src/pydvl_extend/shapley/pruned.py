import abc
import logging
from time import sleep

import numpy as np

from pydvl.utils import ParallelConfig, Utility, init_parallel_backend, running_moments
from pydvl.value import ValuationResult
from pydvl.value.stopping import StoppingCriterion

__all__ = [
    "PruningPolicy",
    "NoPruning",
    "FixedPruning",
    "BootstrapPruning",
    "RelativePruning",
    "pruned_montecarlo_shapley",
]


logger = logging.getLogger(__name__)


class PruningPolicy(abc.ABC):
    """A policy for deciding whether to stop computing marginals in a
    permutation.

    Statistics are kept on the number of calls and prunings as :attr:`n_calls`
    and :attr:`n_prunings` respectively.

    .. todo::
       Because the policy objects are copied to the workers, the statistics
       are not accessible from the
       :class:`~pydvl.value.shapley.actor.ShapleyCoordinator`. We need to add
       methods for this.
    """

    def __init__(self):
        self.n_calls: int = 0
        self.n_prunings: int = 0

    @abc.abstractmethod
    def _check(self, idx: int, score: float) -> bool:
        """Implement the policy."""
        ...

    @abc.abstractmethod
    def reset(self):
        """Reset the policy to a state ready for a new permutation."""
        ...

    def __call__(self, idx: int, score: float) -> bool:
        """Check whether the computation should be interrupted.

        :param idx: Position in the permutation currently being computed.
        :param score: Last utility computed.
        :return: ``True`` if the computation should be interrupted.
        """
        ret = self._check(idx, score)
        self.n_calls += 1
        self.n_prunings += 1 if ret else 0
        return ret


class NoPruning(PruningPolicy):
    """A policy which never interrupts the computation."""

    def _check(self, idx: int, score: float) -> bool:
        return False

    def reset(self):
        pass


class FixedPruning(PruningPolicy):
    """Break a permutation after computing a fixed number of marginals.

    :param u: Utility object with model, data, and scoring function
    :param fraction: Fraction of marginals in a permutation to compute before
        stopping (e.g. 0.5 to compute half of the marginals).
    """

    def __init__(self, u: Utility, fraction: float):
        super().__init__()
        if fraction <= 0 or fraction > 1:
            raise ValueError("fraction must be in (0, 1]")
        self.max_marginals = len(u.data) * fraction
        self.count = 0

    def _check(self, idx: int, score: float) -> bool:
        self.count += 1
        return self.count >= self.max_marginals

    def reset(self):
        self.count = 0


class RelativePruning(PruningPolicy):
    """Break a permutation if the marginal utility is too low.

    This is called "performance tolerance" in :footcite:t:`ghorbani_data_2019`.

    :param u: Utility object with model, data, and scoring function
    :param rtol: Relative tolerance. The permutation is broken if the
        last computed utility is less than ``total_utility * rtol``.
    """

    def __init__(self, u: Utility, atol: float):
        super().__init__()
        self.atol = atol
        logger.info("Computing total utility for permutation pruning.")

    def _check(self, idx: int, score: float) -> bool:
        return np.allclose(score, 0, atol=self.atol)

    def reset(self):
        pass


class BootstrapPruning(PruningPolicy):
    """Break a permutation if the last computed utility is close to the total
    utility, measured as a multiple of the standard deviation of the utilities.

    :param u: Utility object with model, data, and scoring function
    :param n_samples: Number of bootstrap samples to use to compute the variance
        of the utilities.
    :param sigmas: Number of standard deviations to use as a threshold.
    """

    def __init__(self, u: Utility, n_samples: int, sigmas: float = 1):
        super().__init__()
        self.n_samples = n_samples
        logger.info("Computing total utility for permutation pruning.")
        self.total_utility = u(u.data.indices)
        self.count: int = 0
        self.variance: float = 0
        self.mean: float = 0
        self.sigmas: float = sigmas

    def _check(self, idx: int, score: float) -> bool:
        self.mean, self.variance = running_moments(
            self.mean, self.variance, self.count, score
        )
        self.count += 1
        logger.info(
            f"Bootstrap pruning: {self.count} samples, {self.variance:.2f} variance"
        )
        if self.count < self.n_samples:
            return False
        return abs(score - self.total_utility) < float(
            self.sigmas * np.sqrt(self.variance)
        )

    def reset(self):
        self.count = 0
        self.variance = self.mean = 0


def pruned_montecarlo_shapley(
    u: Utility,
    *,
    done: StoppingCriterion,
    pruning: PruningPolicy,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    coordinator_update_period: int = 10,
    worker_update_period: int = 5,
) -> ValuationResult:
    """Monte Carlo approximation to the Shapley value of data points.

    This implements the permutation-based method described in
    :footcite:t:`ghorbani_data_2019`. It is a Monte Carlo estimate of the sum
    over all possible permutations of the index set, with a double stopping
    criterion.

    .. todo::
       Think of how to add Robin-Gelman or some other more principled stopping
       criterion.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation and incrementally compute marginal utilities.
    We stop computing marginals for a given permutation based on a
    :class:`PruningPolicy`. :footcite:t:`ghorbani_data_2019` mention two
    policies: one that stops after a certain fraction of marginals are computed,
    implemented in :class:`FixedPruning`, and one that stops if the last
    computed utility ("score") is close to the total utility using the standard
    deviation of the utility as a measure of proximity, implemented in
    :class:`BootstrapPruning`.

    We keep sampling permutations and updating all shapley values
    until the :class:`StoppingCriterion` returns ``True``.

    :param u: Utility object with model, data, and scoring function
    :param done: Check on the results which decides when to stop
        sampling permutations.
    :param pruning: callable that decides whether to stop computing
        marginals for a given permutation.
    :param n_jobs: number of jobs processing permutations. If None, it will be
        set to :func:`available_cpus`.
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param coordinator_update_period: in seconds. How often to check the
        accumulated results from the workers for convergence.
    :param worker_update_period: interval in seconds between different
        updates to and from the coordinator
    :return: Object with the data values.

    """
    # Avoid circular imports
    from .actor import get_shapley_coordinator, get_shapley_worker

    if config.backend == "sequential":
        raise NotImplementedError(
            "Truncated MonteCarlo Shapley does not work with "
            "the Sequential parallel backend."
        )

    parallel_backend = init_parallel_backend(config)
    n_jobs = parallel_backend.effective_n_jobs(n_jobs)
    u_id = parallel_backend.put(u)

    coordinator = get_shapley_coordinator(config=config, done=done)  # type: ignore

    workers = [
        get_shapley_worker(  # type: ignore
            u=u_id,
            coordinator=coordinator,
            pruning=pruning,
            worker_id=worker_id,
            update_period=worker_update_period,
            config=config,
        )
        for worker_id in range(n_jobs)
    ]
    for worker in workers:
        worker.run(block=False)

    while not coordinator.check_convergence():
        sleep(coordinator_update_period)

    return coordinator.accumulate()
