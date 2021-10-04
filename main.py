import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import read_wav, fft, convolution, write_wav, normalisation, low_pass, band_stop


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
        mod_H_m_dB = 20 * np.log10(mod_H_m)
        w = [2 * math.pi * m / len(mod_H_m) for m in range(len(mod_H_m))]
        deltas.append(gain - np.interp(math.pi / 1000, w, mod_H_m_dB))

    filter_order = range_start + ((np.abs(deltas)).argmin() * range_step)
    print(
        f'closest delta: {deltas[(np.abs(deltas)).argmin()]}dB | DC gain: {mod_H_m_dB[0]}dB| corresponding N: {filter_order}')
    return filter_order


def extract_sin(mod_X_m: np.ndarray, deg_X_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return np.array(m_list), np.array(extracted_mod_X_m), np.array(extracted_deg_X_m)


def build_RIF(N_filter: int, f_c: float, f_e: float, type: str, length) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the RIF filter by:
        1. impulse response -> h[n]
        2. frequency response -> h[m] (h[n] is padded to get a better resolution)
    :param N_filter: filter order
    :param f_c: filter cutoff frequency
    :param f_e: sample rate
    :param type: filter type
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


def build_plots(N_filter: int, mod_X_m: np.ndarray, deg_X_m: np.ndarray, h_n: np.ndarray, mod_H_m: np.ndarray,
                w: np.ndarray) -> None:
    """
     Builds plots from the data extracted from previous functions
    :param N_filter: filter order
    :param mod_X_m: module X[m]
    :param deg_X_m: phase X[m]
    :param h_n: impulse response
    :param mod_H_m: module frequency response
    :param w: omega
    :return: None
    """

    # analysis
    mod_X_m_dB = np.log10(mod_X_m)

    # RIF filter
    mod_H_m_dB = 20 * np.log10(mod_H_m)

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
    plt.stem(range(int((-N_filter / 2) + 1), int(N_filter / 2)), h_n)
    plt.title('réponse impulsionnelle de h[n]')
    plt.xlabel('n')
    plt.ylabel('amplitude')
    plt.figure(3)
    plt.plot(w, mod_H_m_dB)
    plt.title('réponse fréquencielle de H[w]')
    plt.xlabel('w(rad/échantillons)')
    plt.ylabel('amplitude(dB)')


def synthesis(mod_X_m: np.ndarray, phase_X_m: np.ndarray, w: np.ndarray,
              envelop: np.ndarray, k: int) -> np.ndarray:
    """
    add sin and multiply the result with the envelop
    :param mod_X_m: modules of the signal
    :param phase_X_m: phases of the signal
    :param w: omega[m]
    :param envelop: envelop from the convolution
    :param k: k index for different notes
    :return:
    """
    f = (2**(k/12))
    summation = np.zeros(len(envelop))
    for n in tqdm(range(len(envelop))):
        summation[n] = np.sum(np.multiply(mod_X_m, np.sin(np.multiply(f*n, w) + phase_X_m)))

    return np.multiply(summation, envelop)


def build_symphony(mod_X_m, phase_X_m, w_m, envelop, f_e) -> np.ndarray:
    """
    Builds the 5th symphony from Beetoven from the guitar note
    :param mod_X_m: module X[m]
    :param phase_X_m: phase X[m]
    :param w_m: omega w[m]
    :param envelop: envelop obtained from the convolution(h_n, |x[n]|)
    :param f_e: sample rate
    :return: symphony
    """
    symphony = np.array([])
    for k in [-2, -2, -2, -6, 'silence', -4, -4, -4, -7]:
        if k == 'silence':
            symphony = np.append(symphony, np.zeros(int(f_e / 3)))
        else:
            nb_sample = f_e if k in [-6, -7] else int(f_e / 3)
            symphony = np.append(symphony, (synthesis(mod_X_m, phase_X_m, w_m, envelop, k))[:nb_sample])

    plt.figure(4)
    plt.plot(symphony)

    return symphony


if __name__ == '__main__':
    # # find the best N order for the low pass filter
    # N_filter = find_best_N()
    #
    # # extract sines meeting the specs
    # f_e, x_n = read_wav('note_guitare_LAd.wav')
    # window = np.hanning(len(x_n))
    # X_m, mod_X_m, phase_X_m = fft(x_n * window)
    # m, mod_X_m, phase_X_m = extract_sin(mod_X_m, phase_X_m)
    #
    # # filter LA through low_pass to only get the tone
    # f_c_lp = math.pi/1000
    # h_n, mod_H_m = build_RIF(N_filter, f_c_lp, f_e, 'low_pass')
    # w = np.array([normalisation(m, len(mod_H_m)) for m in range(len(mod_H_m))])
    #
    # # make the 5th symphony out of the filtered LA from the guitar
    # envelop = convolution(h_n, np.abs(x_n))
    # symphony = build_symphony(mod_X_m, phase_X_m, w_m, envelop, f_e)
    # write_wav(f_e, symphony, 'note_guitare_LAd_filtered.wav')

    # filter 1000Hz sine in band stop
    f_e, x_n = read_wav('note_basson_plus_sinus_1000_Hz.wav')
    N_band_stop = 6000
    f_c_bs = 1000
    h_n_bs, mod_H_m = build_RIF(6000, f_c_bs, f_e, 'band_stop', len(x_n))

    plt.figure(1)
    plt.stem(range(int(-N_band_stop/2) + 1, int(N_band_stop/2)), h_n_bs)

    # normalise with the ration m/N than use the result to extract f from w = 2*pi*f/f_e
    f = np.array([(normalisation(m, len(x_n))*f_e)/(2 * math.pi) for m in range(len(x_n))])

    plt.figure(3)
    plt.plot(f, 20*np.log10(mod_H_m))

    y_n = convolution(h_n_bs, x_n)
    write_wav(f_e, y_n, 'basson_filtered.wav')

    plt.figure(2)
    plt.stem(y_n)
    plt.show()

    # plot parameters
    # build_plots(N_filter, mod_X_m, phase_X_m, h_n, mod_H_m, w)
