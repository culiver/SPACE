import abc
import logging
import warnings
from typing import Iterable, Optional

import numpy as np

from pydvl.utils.config import ParallelConfig
from pydvl.utils.numeric import random_powerset
from pydvl.utils.parallel import MapReduceJob
from pydvl.utils.parallel.backend import effective_n_jobs
from pydvl.utils.progress import maybe_progress
from pydvl.utils.utility import Utility
from pydvl.value.least_core.common import LeastCoreProblem, lc_solve_problem
from pydvl.value.result import ValuationResult

logger = logging.getLogger(__name__)


__all__ = [
    "montecarlo_least_core",
    "mclc_prepare_problem",
]


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

def _pruned_montecarlo_least_core(
    u: Utility, n_iterations: int, *, progress: bool = False, job_id: int = 1, pruning: PruningPolicy
) -> LeastCoreProblem:
    """Computes utility values and the Least Core upper bound matrix for a given number of iterations.

    :param u: Utility object with model, data, and scoring function
    :param n_iterations: total number of iterations to use
    :param progress: If True, shows a tqdm progress bar
    :param job_id: Integer id used to determine the position of the progress bar
    :return:
    """
    n = len(u.data)

    # Revised Version
    utility_values = np.zeros(n_iterations * n)
    A_lb = np.zeros((n_iterations * n, n))

    for iter in range(n_iterations):
        permutation = np.random.permutation(u.data.indices)
        permutation_done = False

        pruning.reset()
        for i in range(len(permutation)):
            if permutation_done:
                score = 0
            else:
                score = u(permutation[: -i])
            indices = np.zeros(n, dtype=bool)
            indices[list(permutation[: -i])] = True
            A_lb[iter*n + i, indices] = 1
            utility_values[iter*n + i] = score

            if not permutation_done and pruning(i, score):
                permutation_done = True

    return LeastCoreProblem(utility_values, A_lb)


def _reduce_func(results: Iterable[LeastCoreProblem]) -> LeastCoreProblem:
    """Combines the results from different parallel runs of
    :func:`_pruned_montecarlo_least_core`"""
    utility_values_list, A_lb_list = zip(*results)
    utility_values = np.concatenate(utility_values_list)
    A_lb = np.concatenate(A_lb_list)
    return LeastCoreProblem(utility_values, A_lb)


def pruned_montecarlo_least_core(
    u: Utility,
    n_iterations: int,
    pruning: PruningPolicy,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    options: Optional[dict] = None,
    progress: bool = False,
) -> ValuationResult:
    r"""Computes approximate Least Core values using a Monte Carlo approach.

    $$
    \begin{array}{lll}
    \text{minimize} & \displaystyle{e} & \\
    \text{subject to} & \displaystyle\sum_{i\in N} x_{i} = v(N) & \\
    & \displaystyle\sum_{i\in S} x_{i} + e \geq v(S) & ,
    \forall S \in \{S_1, S_2, \dots, S_m \overset{\mathrm{iid}}{\sim} U(2^N) \}
    \end{array}
    $$

    Where:

    * $U(2^N)$ is the uniform distribution over the powerset of $N$.
    * $m$ is the number of subsets that will be sampled and whose utility will
      be computed and used to compute the data values.

    :param u: Utility object with model, data, and scoring function
    :param n_iterations: total number of iterations to use
    :param n_jobs: number of jobs across which to distribute the computation
    :param config: Object configuring parallel computation, with cluster
        address, number of cpus, etc.
    :param options: Keyword arguments that will be used to select a solver
        and to configure it. Refer to the following page for all possible options:
        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
    :param progress: If True, shows a tqdm progress bar
    :return: Object with the data values and the least core value.
    """
    problem = mclc_prepare_problem(
        u, n_iterations, n_jobs=n_jobs, config=config, progress=progress, pruning=pruning
    )
    return lc_solve_problem(
        problem, u=u, algorithm="montecarlo_least_core", **(options or {})
    )


def mclc_prepare_problem(
    u: Utility,
    n_iterations: int,
    *,
    n_jobs: int = 1,
    config: ParallelConfig = ParallelConfig(),
    progress: bool = False,
    pruning: PruningPolicy = NoPruning(),
) -> LeastCoreProblem:
    """Prepares a linear problem by sampling subsets of the data.
    Use this to separate the problem preparation from the solving with
    :func:`~pydvl.value.least_core.common.lc_solve_problem`. Useful for
    parallel execution of multiple experiments.

    See :func:`montecarlo_least_core` for argument descriptions.
    """
    n = len(u.data)

    if n_iterations < n:
        raise ValueError(
            "Number of iterations should be greater than the size of the dataset"
        )

    if n_iterations > 2**n:
        warnings.warn(
            f"Passed n_iterations is greater than the number subsets! "
            f"Setting it to 2^{n}",
            RuntimeWarning,
        )
        n_iterations = 2**n

    iterations_per_job = max(1, n_iterations // effective_n_jobs(n_jobs, config))

    map_reduce_job: MapReduceJob["Utility", "LeastCoreProblem"] = MapReduceJob(
        inputs=u,
        map_func=_pruned_montecarlo_least_core,
        reduce_func=_reduce_func,
        map_kwargs=dict(n_iterations=iterations_per_job, progress=progress, pruning=pruning),
        n_jobs=n_jobs,
        config=config,
    )

    return map_reduce_job()
