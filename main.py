import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import read_wav, fft


def find_best_N() -> int:
    """
    modify the range to search for the best N filter order
    specs:
            gain: -3dB at pi/1000
            max range: 2pi/N+1 > pi/1000 -> N < 2001
            DC gain: 0 dB

    :return: N filter order matching the specs
    """
    gain = -3
    range_start = 850
    range_stop = 900
    range_step = 1
    deltas = []
    for N in tqdm(range(range_start, range_stop, range_step)):
        h_n, mod_H_m = build_RIF(N)
        mod_H_m_dB = 20*np.log10(mod_H_m)
        w = [2 * math.pi * m / len(mod_H_m) for m in range(len(mod_H_m))]
        deltas.append(gain - np.interp(math.pi / 1000, w, mod_H_m_dB))

    filter_order = range_start + ((np.abs(deltas)).argmin() * range_step)
    print(f'closest delta: {deltas[(np.abs(deltas)).argmin()]}dB | DC gain: {mod_H_m_dB[0]}dB| corresponding N: {filter_order}')
    return filter_order


def extract_sin(mod_X_m: list, deg_X_m: list) -> Tuple[list, list, list]:
    """
    requirements to choose a frequency: 10% max amplitude, amplitude[m] > amplitude[m+1] et amplitude[m] > amplitude[m-1]
    :param mod_X_m: module of X[m]
    :param deg_X_m: phase of X[m]
    :return: data from the input meeting the requirements
    """
    req_max_amplitude = 0.1 * max(mod_X_m)

    extracted_mod_X_m = []
    extracted_deg_X_m = []
    m_list = []
    for i, amplitude in enumerate(mod_X_m):
        # if the current amplitude meets the 10% of max requirement
        if amplitude >= req_max_amplitude:
            # we don't want to check the last one because it's going to throw
            if i != len(mod_X_m) - 1:
                # if the current is bigger than the one before and after
                if mod_X_m[i - 1] < amplitude > mod_X_m[i + 1]:
                    extracted_deg_X_m.append(deg_X_m[i])
                    extracted_mod_X_m.append(amplitude)
                    m_list.append(i)
    return m_list, extracted_mod_X_m, extracted_deg_X_m


def build_RIF(N_filter: int) -> Tuple[list, list]:
    """
    Builds the RIF filter by:
        1. impulse response -> h[n]
        2. frequency response -> h[m] (h[n] is padded to get a better resolution)
    :param N_filter: filter order
    :return: the impulse response and module of the frequency response
    """
    # w_c = 2*pi*m/N -> m = w_c*N/2*pi where w_c is pi/1000
    m = int((math.pi / 1000) * N_filter / (2 * math.pi))
    k = 2 * m + 1

    # impulse response
    h_n = [(1 / N_filter) * (math.sin(math.pi * n * k / N_filter) / math.sin(math.pi * n / N_filter)) for n in range(int(-N_filter/2)+1, int(N_filter/2)) if n != 0]
    h_n.insert(math.ceil(len(h_n)/2), k / N_filter)

    # frequency response
    H_m, mod_H_m, _ = fft(np.concatenate((h_n, np.zeros(160000-len(h_n)))))
    return h_n, mod_H_m


def build_plots(N_filter: int):
    """
    Builds plots from the data extracted from previous functions
    :param N_filter: filter order
    """

    # analysis
    f_e, x_n = read_wav()
    X_m, mod_X_m, deg_X_m = fft(x_n)
    m, mod_X_m, deg_X_m = extract_sin(mod_X_m, deg_X_m)
    mod_X_m_dB = np.log10(mod_X_m)

    # RIF filter
    h_n, mod_H_m = build_RIF(N_filter)
    mod_H_m_dB = 20 * np.log10(mod_H_m)
    w = [2 * math.pi * m / len(mod_H_m) for m in range(len(mod_H_m))]

    # analysis plots
    fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(10, 10))
    ax[0].plot(X_m)
    ax[0].set_title('X_m')
    ax[1].stem(m, mod_X_m)
    ax[1].set_title('module X_m')
    ax[1].set_ylabel('amplitude')
    ax[1].set_xlabel('m')
    ax[2].stem(m, mod_X_m_dB)
    ax[2].set_title('module X_m')
    ax[2].set_ylabel('amplitude(dB)')
    ax[2].set_xlabel('m')
    ax[3].stem(m, deg_X_m)
    ax[3].set_title('phase X_m')
    ax[3].set_ylabel('phase(deg)')
    ax[3].set_xlabel('m')
    plt.tight_layout(pad=3.0)

    # RIF filter plots
    plt.figure(2)
    plt.stem(range(int((-N_filter/2)+1), int(N_filter/2)), h_n)
    plt.title('réponse impulsionnelle de h[n]')
    plt.xlabel('n')
    plt.ylabel('amplitude')
    plt.figure(3)
    plt.plot(w, mod_H_m_dB)
    plt.title('réponse fréquencielle de H[w]')
    plt.xlabel('w(rad/échantillons)')
    plt.ylabel('amplitude(dB)')
    plt.show()


if __name__ == '__main__':
    N_filter = find_best_N()
    build_plots(N_filter)
