"""Utility helpers: sampling, GMM fitting and simple metrics."""
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance

def sample_from_mixture(pis, means, stds, n):
    K = pis.shape[0]
    comps = np.random.choice(K, size=n, p=pis)
    samples = np.zeros((n, means.shape[1]), dtype=np.float32)
    for i, c in enumerate(comps):
        samples[i] = np.random.normal(loc=means[c], scale=stds[c])
    return samples

def fit_ep_gmm(X, max_components=6, random_state=0):
    best = None
    best_bic = float('inf')
    for k in range(1, max_components+1):
        gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=random_state)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best = gmm
    return dict(pis=best.weights_, means=best.means_, stds=np.sqrt(best.covariances_), model=best)

def episode_wasserstein_2d(p_true, p_pred, n_samples=2000):
    X_true = sample_from_mixture(p_true['pis'], p_true['means'], p_true['stds'], n_samples)
    X_pred = sample_from_mixture(p_pred['pis'], p_pred['means'], p_pred['stds'], n_samples)
    wd0 = wasserstein_distance(X_true[:,0], X_pred[:,0])
    wd1 = wasserstein_distance(X_true[:,1], X_pred[:,1])
    return 0.5 * (wd0 + wd1)
