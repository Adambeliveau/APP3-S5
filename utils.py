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
    window = np.hanning(160000)
    output_m = np.fft.fft(input_n*window)
    return output_m, np.abs(output_m), np.angle(output_m)


def convolution(X_m: np.ndarray, H_m: np.ndarray) -> np.ndarray:
    """
    convolution between the signal and the frequency response
    :param X_m: signal
    :param H_m: frequency response
    :return: convolution
    """
    return np.convolve(X_m, H_m)
