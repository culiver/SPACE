from enum import Enum


class ShapleyMode(str, Enum):
    """Supported algorithms for the computation of Shapley values.

    .. todo::
       Make algorithms register themselves here.
    """

    CombinatorialExact = "combinatorial_exact"
    CombinatorialMontecarlo = "combinatorial_montecarlo"
    GroupTesting = "group_testing"
    KNN = "knn"
    Owen = "owen"
    OwenAntithetic = "owen_antithetic"
    PermutationExact = "permutation_exact"
    PermutationMontecarlo = "permutation_montecarlo"
    TruncatedMontecarlo = "truncated_montecarlo"
    PrunedMontecarlo = "pruned_montecarlo"
