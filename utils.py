from typing import Tuple

import numpy as np
from scipy.io import wavfile as wf


def read_wav() -> Tuple[int, np.ndarray]:
    """
    read the .wav file
    :return: samplerate, data
    """
    return wf.read('note_guitare_LAd.wav')


def write_wav(samplerate: int, signal: np.ndarray):
    """
    write the synthesised signal in a wav file
    :param samplerate: samplerate for the wav file
    :param signal: signal to be saved
    :return:
    """
    wf.write('note_guitare_LAd_filtered.wav', samplerate, signal)


def fft(input_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    fft the input
    :param input_n:
    :return: transformed input, module of output, phase of output
    """
    output_m = np.fft.fft(input_n)
    return output_m, np.abs(output_m), np.angle(output_m)


def convolution(h_n: np.ndarray, mod_x_n: np.ndarray) -> np.ndarray:
    """
    convolution between the signal and the frequency response
    :param h_n: impulse response
    :param mod_x_n: module x[n]
    :return: convolution
    """
    return np.convolve(h_n, mod_x_n)


def normalisation(m: int, N: int) -> float:
    """
    time domain to frequency domain conversion
    :param m: samples(time domain)
    :param N: number of samples
    :return: samples(frequency domain) -> omega
    """
    return 2 * np.pi * m / N
