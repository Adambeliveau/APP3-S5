import math
from typing import Tuple

import numpy as np
from scipy.io import wavfile as wf
import matplotlib.pyplot as plt


def read_wav(filename: str) -> Tuple[int, np.ndarray]:
    """
    read the .wav file
    :param: name of the input file
    :return: samplerate, data
    """
    return wf.read(filename)


def write_wav(samplerate: int, signal: np.ndarray, filename: str):
    """
    write the synthesised signal in a wav file
    :param samplerate: samplerate for the wav file
    :param signal: signal to be saved
    :param filename: name for the output file
    :return:
    """
    wf.write(filename, samplerate, signal)


def fft(input_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    fft the input
    :param input_n:
    :return: transformed input, module of output, phase of output
    """
    output_m = np.fft.fft(input_n)
    return output_m, np.abs(output_m), np.angle(output_m)


def convolution(h_n: np.ndarray, x_n: np.ndarray) -> np.ndarray:
    """
    convolution between the signal and the frequency response
    :param h_n: impulse response
    :param x_n: x[n]
    :return: convolution
    """
    return np.convolve(h_n, x_n)


def normalisation(m: float, N: float) -> float:
    """
    time domain to frequency domain conversion
    :param m: samples(time domain)
    :param N: number of samples
    :return: samples(frequency domain) -> omega

    ** you can use the ratio f_c/f_e instead of m/N
    """
    w = 2 * np.pi * m / N
    return w if w < math.pi else w - (2 * math.pi)


def low_pass(N_filter: int, f_c: float, f_e: float) -> np.ndarray:
    # w_c = 2*pi*m/N -> m = w_c*N/2*pi where w_c
    w_c = normalisation(f_c, f_e)
    m = int(w_c * N_filter / (2 * math.pi))
    k = 2 * m + 1

    h_n = np.array([(1 / N_filter) * (math.sin(math.pi * n * k / N_filter) / math.sin(math.pi * n / N_filter))
                    for n
                    in range(int(-N_filter / 2) + 1, int(N_filter / 2))
                    if n != 0])

    h_n = np.insert(h_n, math.ceil(len(h_n) / 2), k / N_filter)

    return h_n


def band_stop(N_filter: int, f_c: float, f_e: float) -> np.ndarray:
    # 2f_c_1 = 1040 - 960 -> f_c_1 = 40
    f_c_lp = 40
    h_n_lp = low_pass(N_filter, f_c_lp, f_e)

    w_c = normalisation(f_c, f_e)

    delta = np.zeros(len(h_n_lp))
    delta[0] = 1
    h_n = np.array([delta[n] - 2*h_n_lp[n]*np.cos(w_c*n)
                   for n
                   in range(int(-N_filter / 2) + 1, int(N_filter / 2))])

    return h_n


