import math
import numpy as np
from scipy import linalg
from scipy.special import softmax

def stable_sigmoid(x):

    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
            representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
            'adding %s to diagonal of cov estimates') % eps
        # print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

class Prototype_set():
    def __init__(self, clientPrototypes):
        self.prototypes = clientPrototypes
        self.indices = np.arange(len(self.prototypes))
        self.data_names = self.indices
    def __len__(self):    
        return len(self.prototypes)

class Utility_Func_cosine():
    def __init__(
        self,
        serverPrototype,
        clientPrototypes,
        client_data_nums,
        u_trans=False,
        k=10,
        T=1.0,
    ):
        self.serverPrototype = serverPrototype
        self.data = Prototype_set(clientPrototypes)
        self.client_data_nums = client_data_nums
        self.u_trans = u_trans
        self.k = k
        if math.isnan(T):
            full_coalition_value = self.scorer(self.data.indices)
            self.T = full_coalition_value
        else:
            self.T = T

    def __call__(self, indices):
        utility: float = self._utility(indices)
        return utility

    def scorer(self, indices, metric='cosine'):
        epsilon = 1e-8
        indices = tuple(indices)

        # Merge selected clients
        #  weight: (num_user, num_class) or (num_user, 2, num_class)
        weights = self.client_data_nums[np.array(indices)] / np.maximum(self.client_data_nums[np.array(indices)].sum(axis=0, keepdims=True), epsilon)
        weights = weights[..., np.newaxis]
        clientPrototype = (self.data.prototypes[indices, :] * weights).sum(axis=0)
        # Normalization
        serverPrototype = self.serverPrototype / np.linalg.norm(self.serverPrototype, axis=1, keepdims=True)
        clientPrototype = clientPrototype / np.maximum(np.linalg.norm(clientPrototype, axis=-1, keepdims=True), epsilon)

        distMatrix = serverPrototype @ clientPrototype.T
        distMatrix = softmax(distMatrix, axis=-1)
        score = np.diagonal(distMatrix).copy().mean()
        
        

        return score

    def _utility(self, indices) -> float:
        if len(indices) == 0:
            return 0.0
        score = self.scorer(indices)
        if self.u_trans:
            score = stable_sigmoid((score-self.T) * self.k) if (score != self.T) else 0.5    
        return score

class Distribution_set():
    def __init__(self, clientMus, clientSigmas):
        self.Mus = clientMus
        self.Sigmas = clientSigmas
        self.indices = np.arange(len(self.Mus))
        self.data_names = self.indices
    def __len__(self):    
        return len(self.indices)

class Utility_Func_fid():
    def __init__(
        self,
        serverMu,
        serverSigma,
        clientMus,
        clientSigmas,
        client_data_nums,
        u_trans=False,
        k=10,
        T=1.0,
    ):
        self.serverMu = serverMu
        self.serverSigma = serverSigma
        self.data = Distribution_set(clientMus, clientSigmas)
        self.client_data_nums = client_data_nums
        self.u_trans = u_trans
        self.k = k
        if math.isnan(T):
            full_coalition_value = self.scorer(self.data.indices)
            self.T = full_coalition_value
        else:
            self.T = T
        
    def __call__(self, indices):
        utility: float = self._utility(indices)
        return utility
    
    

    def scorer(self, indices, metric='cosine'):
        epsilon = 1e-8
        indices = tuple(indices)

        # Merge selected clients
        #  weight: (num_user, num_class) or (num_user, 2, num_class)
        weights = self.client_data_nums[np.array(indices)] / np.maximum(self.client_data_nums[np.array(indices)].sum(axis=0, keepdims=True), epsilon)
        weights = weights[..., np.newaxis]
        selected_mu = self.data.Mus[indices, :]

        # New Mu
        clientMu = (selected_mu * weights).sum(axis=0)
        # New Sigma = Weighted Sigma + Weighted Mu @ Mu^T - newMu @ newMu^T
        dyadic_mu = np.einsum('ijk,ijm->ijkm', selected_mu, selected_mu)
        clientSigma = (self.data.Sigmas[indices, :] * weights[..., np.newaxis]).sum(axis=0) + (dyadic_mu * weights[..., np.newaxis]).sum(0) - np.einsum('ij,ik->ijk', clientMu, clientMu)

        score = []
        for i in range(self.serverMu.shape[0]):
            distance = []
            for j in range(self.serverMu.shape[0]):
                if np.all(clientMu[j] == 0):
                    distance.append(np.inf)
                else:
                    distance.append(calculate_frechet_distance(self.serverMu[i], self.serverSigma[i], clientMu[j], clientSigma[j]))
            similarity = np.reciprocal(np.array(distance))
            similarity = softmax(similarity)
            score.append(similarity[i])
        score = np.array(score).mean()

        
        
        return score

    def _utility(self, indices) -> float:
        if len(indices) == 0:
            return 0.0
        score = self.scorer(indices)
        if self.u_trans:
            score = stable_sigmoid((score-self.T) * self.k) if (score != self.T) else 0.5    
        return score

class RealShap_set():
    def __init__(self, subsets_info):
        self.subsets_info = subsets_info
        num_users = self.subsets_info['num_users']
        self.indices = np.arange(num_users)
        self.data_names = self.indices
    def __len__(self):    
        return len(self.indices)

class Utility_Func_RealShap():
    def __init__(
        self,
        subsets_info,
        u_trans=False,
        k=10,
        T=1.0,
    ):
        self.subsets_info = subsets_info
        self.data = RealShap_set(subsets_info)
        self.u_trans = u_trans
        self.k = k
        if math.isnan(T):
            full_coalition_value = self.scorer(self.data.indices)
            self.T = full_coalition_value
        else:
            self.T = T
        
    def __call__(self, indices):
        utility: float = self._utility(indices)
        return utility

    def scorer(self, indices, metric='cosine'):
        epsilon = 1e-8
        indices = [str(i) for i in sorted(list(indices))]
        query = '+'.join(indices)
        score = self.subsets_info[query]
        

        return score

    def _utility(self, indices) -> float:
        if len(indices) == 0:
            return 0.0
        score = self.scorer(indices)
        if self.u_trans:
            score = stable_sigmoid((score-self.T) * self.k) if (score != self.T) else 0.5    
        return score