import mne
import numpy as np
from scipy.linalg import (eig, lstsq)


def ssd(raw, method='filter', keepN = None):

    fs = raw.info['sfreq']
    x = np.real(raw.get_data())
    if method == 'filter':
        xs = mne.filter.filter_data(x, sfreq=fs, l_freq=8, h_freq=12, n_jobs=2, method='iir', phase='zero-double')
        y1 = mne.filter.filter_data(x, sfreq=fs, l_freq=6, h_freq=14, n_jobs=2, method='iir', phase='zero-double')
        xn = mne.filter.filter_data(y1, sfreq=fs, l_freq=13, h_freq=7, n_jobs=2, method='iir', phase='zero-double')
        C_s = np.cov(xs)
        C_n = np.cov(xn)
    #elif method == 'fft':

    D, V = eig(C_s)
    D, V = np.real(D), np.real(V)

    ev_sorted = np.sort(D)
    sort_idx = np.argsort(D)
    sort_idx = sort_idx[::-1]
    ev_sorted = ev_sorted[::-1]
    V = V[:, sort_idx]
    tol = ev_sorted[0] * 10 ** -6
    r = np.sum(ev_sorted > tol)

    if r < x.shape[0]:
        lambda2 = ev_sorted[0:r].reshape((1, r))
        M = V[:, 0:r] * (1 / np.sqrt(lambda2))
    else:
        M = np.eye(C_s.shape)

    C_s_r = np.dot(np.dot(M.T, C_s), M)
    C_n_r = np.dot(np.dot(M.T, C_n), M)

    D, W = eig(C_s_r, C_s_r + C_n_r)

    sort_idx = np.argsort(D)
    sort_idx = sort_idx[::-1]
    W = W[:, sort_idx]
    W = W[:, 0:keepN] if keepN is not None else W

    W = np.dot(M, W)
    A = lstsq(np.dot(np.dot(W.T, C_s), W), np.dot(W.T, C_s))[0]
    A = A.T
    return A, W


