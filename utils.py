import math
from typing import Tuple

import numpy as np
from scipy.io import wavfile as wf


def read_wav(filename: str) -> Tuple[int, np.ndarray]:
    """
    read the .wav file
    :param: name of the input file
    :return: samplerate, data
    """
    return wf.read(filename)


def write_wav(samplerate: int, signal: np.ndarray, filename: str) -> None:
    """
    write the synthesised signal in a wav file
    :param samplerate: samplerate for the wav file
    :param signal: signal to be saved
    :param filename: name for the output file
    :return:
    """
    wf.write(filename, samplerate, signal)


def extract_sin(mod_X_m: np.ndarray, phase_X_m: np.ndarray, f_e: float, w_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    requirements to choose a frequency: 10% max amplitude, amplitude[m] > amplitude[m+1] et amplitude[m] > amplitude[m-1]
    :param mod_X_m: module of X[m]
    :param phase_X_m: phase of X[m]
    :param f_e: sample rate
    :param w_list: list of normalised samples
    :return: data from the input meeting the requirements
    """
    f_fundamental = 466.2
    w_fundamental = normalisation(f_fundamental, f_e)

    index_harmonics = np.array([], dtype=int)
    for i in range(int(max(w_list)/w_fundamental)):
        w = w_fundamental*i
        index_float = (w * len(w_list)) / max(w_list)
        choice = np.array([mod_X_m[math.ceil(index_float)], mod_X_m[math.floor(index_float)]]).argmax()
        index_harmonics = np.append(index_harmonics, math.floor(index_float) if choice else math.ceil(index_float))

    harmonics_data = [(i, mod_X_m[i], phase_X_m[i]) for i in index_harmonics]
    sorted_modules = sorted(harmonics_data, key=lambda x: x[1], reverse=True)[:32]
    m_index, mod_X_m, phase_X_m = zip(*sorted_modules)

    return np.array(m_index), np.array(mod_X_m), np.array(phase_X_m)


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


def normalisation(m: float, N: float, shifted=False) -> float:
    """
    time domain to frequency domain conversion
    :param m: samples(time domain)
    :param N: number of samples
    :param shifted: when true, w will be between -pi and pi
    :return: samples(frequency domain) -> omega

    ** you can use the ratio f_c/f_e instead of m/N
    """
    w = 2 * np.pi * m / N
    if shifted:
        return w
    else:
        return w if w < math.pi else w - (2 * math.pi)




