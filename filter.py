import math
from typing import Tuple

import numpy as np
from tqdm import tqdm

from utils import normalisation, fft


def find_best_N(f_c: float, f_e: float, N_ech: int) -> int:
    """
    modify the range to search for the best N filter order
    specs:
            gain: -3dB at pi/1000
            max range: 2pi/N+1 > pi/1000 -> N < 2001
            DC gain: 0 dB

    :param f_c: cutoff frequency for the filter
    :param f_e: sample rate
    :param N_ech: Number of samples
    :return: N filter order matching the specs
    """
    gain = -3
    range_start = 850
    range_stop = 900
    range_step = 1
    deltas = []
    for N in tqdm(range(range_start, range_stop, range_step), desc='finding the best N'):
        h_n, mod_H_m = build_RIF(N, f_c, f_e, 'low_pass', N_ech)
        mod_H_m_dB = 20 * np.log10(mod_H_m)
        w = [2 * math.pi * m / len(mod_H_m) for m in range(len(mod_H_m))]
        deltas.append(gain - np.interp(math.pi / 1000, w, mod_H_m_dB))

    filter_order = range_start + ((np.abs(deltas)).argmin() * range_step)
    print(
        f'closest delta: {deltas[(np.abs(deltas)).argmin()]}dB | DC gain: {mod_H_m_dB[0]}dB| corresponding N: {filter_order}')
    return filter_order


def build_RIF(N_filter: int, f_c: float, f_e: float, type: str, length) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the RIF filter by:
        1. impulse response -> h[n]
        2. frequency response -> H[m] (h[n] is padded to get a better resolution)
    :param N_filter: filter order
    :param f_c: filter cutoff frequency
    :param f_e: sample rate
    :param type: filter type
    :param length: final length of H[m] (used when padding h[n])
    :return: h_n, H_m, mod_H_m
    """

    # impulse response
    impulse_switcher = {
        'low_pass': low_pass,
        'band_stop': band_stop
    }
    h_n = impulse_switcher[type](N_filter, f_c, f_e)

    # frequency response
    H_m, mod_H_m, _ = fft(np.concatenate((h_n, np.zeros(length - len(h_n)))))
    return h_n, mod_H_m


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
