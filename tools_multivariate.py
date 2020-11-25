import numpy as np
import mne
from scipy.linalg import (eig, lstsq, inv, sqrtm, svd, norm)
from numpy import pi
from numpy.linalg import multi_dot
# ======================================================================================================================


def ssd(data, fs, fc, method='filter', n_keep=None):
    """
    The function implimenting Spatio-Spectral Decomposition

    Nikulin,  V.V.,  Nolte,  G.,  Curio,  G.,  2011.    A  novel  method  for  reliableand  fast  extraction  of
    neuronal EEG/MEG  oscillations  on  the  basis  of spatio-spectral decomposition.  NeuroImage 55, 1528–1535.
    doi:10.1016/j.neuroimage.2011.01.057

    parameters
    -----------
    :param data: numpy array [channel x time]:
    :param fs: integer
               sampling frequency
    :param freqs: numpy ndarray [3 x 2]
    :param method: string
                   'time' or 'fft'
    :param n_keep:
    :return:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    (c) Mina Jamshidi Idaji, Nov. 2020
    minajamshidi91@gmail.com
    github.com/minajamshidi

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
    IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    from scipy.signal import butter, filtfilt
    from tools_general import hilbert_

    x = np.real(data)
    iir_params = dict(order=2, ftype='butter')

    if method == 'filter':
        freqs = np.array([[fc - 1, fc + 1], [fc - 3, fc + 3], [fc - 2, fc + 2]])
        b_s, a_s = butter(2, freqs[0, :] / fs * 2, 'bandpass')
        xs = hilbert_(filtfilt(b_s, a_s, x))
        b_n1, a_n1 = butter(2, freqs[1, :] / fs * 2, 'bandpass')
        y1 = filtfilt(b_n1, a_n1, x)
        b_n2, a_n2 = butter(2, freqs[2, :] / fs * 2, 'bandstop')
        xn = hilbert_(filtfilt(b_n2, a_n2, y1))

        cov_sig = np.real(np.cov(xs))
        cov_noise = np.real(np.cov(xn))
    #elif method == 'fft':

    eig_val, eig_vec = eig(cov_sig)
    eig_val, eig_vec = np.real(eig_val), np.real(eig_vec)

    ev_sorted = np.sort(eig_val)[::-1]
    sort_idx = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:, sort_idx]
    tol = ev_sorted[0] * 10 ** -6
    r = np.sum(ev_sorted > tol)

    if r < x.shape[0]:
        lambda2 = ev_sorted[0:r].reshape((1, r))
        whitening_mat = eig_vec[:, 0:r] * (1 / np.sqrt(lambda2))
    else:
        whitening_mat = np.eye(cov_sig.shape[0], cov_sig.shape[1])

    cov_s_r = np.dot(np.dot(whitening_mat.T, cov_sig), whitening_mat)
    cov_n_r = np.dot(np.dot(whitening_mat.T, cov_noise), whitening_mat)

    eig_val, spatial_filter = eig(cov_s_r, cov_s_r + cov_n_r)

    sort_idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sort_idx]
    spatial_filter = spatial_filter[:, sort_idx]
    spatial_filter = spatial_filter[:, 0:n_keep] if n_keep is not None else spatial_filter

    spatial_filter = np.dot(whitening_mat, spatial_filter)
    spatial_pattern = lstsq(np.linalg.multi_dot([spatial_filter.T, cov_sig, spatial_filter]), np.dot(spatial_filter.T, cov_sig))[0]
    spatial_pattern = spatial_pattern.T
    x_ssd = np.dot(spatial_filter.T, xs)
    return x_ssd, spatial_pattern, spatial_filter, eig_val
# ======================================================================================================================


def hpmax(data, fs, fc, H, freqs=None, method='filter', n_keep=None):
    """
    The function implimenting Hamornic-Power Maximization (HPMax)

    parameters
    -----------
    :param data: numpy ndarray [channel x time]:
    :param fs: integer
              sampling rate
    :param freqs: numpy ndarray [3 x 2 x2]
    :param method:
    :param n_keep:
    :retur
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    (c) Mina Jamshidi Idaji, Nov. 2020
    minajamshidi91@gmail.com
    github.com/minajamshidi

    * In case of using this code in a publication, please cite the following paper:

    Sarah Bartz, Forooz Shahbazi Avarvand, Gregor Leicht, Guido Nolte,
    "Analyzing the waveshape of brain oscillations with bicoherence", NeuroImage, Volume 188, 2019,
    https://doi.org/10.1016/j.neuroimage.2018.11.045.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
    OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
    IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
    OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    from scipy.signal import butter, filtfilt
    from tools_general import hilbert_

    x = np.real(data)
    cov_sig = 0
    cov_noise = 0
    if method == 'filter':
        for i_h, h in enumerate(H):
            if freqs is not None:
                freqs_h = freqs[:, :, i_h]
            else:
                freqs_h = np.array([[fc * h - 1, fc * h + 1], [fc * h - 3, fc * h + 3], [fc * h - 2, fc * h + 2]])
            b_s, a_s = butter(2, freqs_h[0, :] / fs * 2, 'bandpass')
            xs = hilbert_(filtfilt(b_s, a_s, x))
            b_n1, a_n1 = butter(2, freqs_h[1, :] / fs * 2, 'bandpass')
            y1 = filtfilt(b_n1, a_n1, x)
            b_n2, a_n2 = butter(2, freqs_h[2, :] / fs * 2, 'bandstop')
            xn = hilbert_(filtfilt(b_n2, a_n2, y1))
            cov_sig += np.real(np.cov(xs))
            cov_noise += np.real(np.cov(xn))

    # elif method == 'fft':

    eig_val, eig_vec = eig(cov_sig)
    eig_val, eig_vec = np.real(eig_val), np.real(eig_vec)

    ev_sorted = np.sort(eig_val)[::-1]
    sort_idx = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:, sort_idx]
    tol = ev_sorted[0] * 10 ** -6
    r = np.sum(ev_sorted > tol)

    if r < x.shape[0]:
        lambda2 = ev_sorted[0:r].reshape((1, r))
        whitening_mat = eig_vec[:, 0:r] * (1 / np.sqrt(lambda2))
    else:
        whitening_mat = np.eye(cov_sig.shape[0], cov_sig.shape[1])

    cov_s_r = np.dot(np.dot(whitening_mat.T, cov_sig), whitening_mat)
    cov_n_r = np.dot(np.dot(whitening_mat.T, cov_noise), whitening_mat)

    eig_val, spatial_filter = eig(cov_s_r, cov_s_r + cov_n_r)

    sort_idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sort_idx]
    spatial_filter = spatial_filter[:, sort_idx]
    spatial_filter = spatial_filter[:, 0:n_keep] if n_keep is not None else spatial_filter

    spatial_filter = np.dot(whitening_mat, spatial_filter)
    spatial_pattern = lstsq(np.linalg.multi_dot([spatial_filter.T, cov_sig, spatial_filter]), np.dot(spatial_filter.T, cov_sig))[0]
    spatial_pattern = spatial_pattern.T
    return spatial_pattern, spatial_filter, eig_val
# ======================================================================================================================

