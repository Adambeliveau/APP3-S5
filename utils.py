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




