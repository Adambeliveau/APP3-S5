import math

import numpy as np

from utils import normalisation


def low_pass(N_filter: int, f_c: float, f_e: float) -> np.ndarray:
    # w_c = 2*pi*m/N -> m = w_c*N/2*pi where w_c
    w_c = normalisation(f_c, f_e)
    m = int(w_c * N_filter / (2 * math.pi))
    k = 2 * m + 1

    h_n = np.array([(1 / N_filter) * (math.sin(math.pi * n * k / N_filter) / math.sin(math.pi * n / N_filter))
                    if n != 0
                    else k/N_filter
                    for n
                    in range(int(-N_filter / 2) + 1, int(N_filter / 2))
                    ])

    return h_n


def band_stop(N_filter: int, f_c: float, f_e: float) -> np.ndarray:
    # 2f_c_1 = 1040 - 960 -> f_c_1 = 40
    f_c_lp = 40
    h_n_lp = low_pass(N_filter, f_c_lp, f_e)

    w_c = normalisation(f_c, f_e)

    delta = np.zeros(len(h_n_lp))
    delta[int(len(h_n_lp)/2)] = 1
    h_n = np.array([delta[n] - 2*h_n_lp[n]*np.cos(w_c*n)
                   for n
                   in range(len(h_n_lp))])

    return h_n
