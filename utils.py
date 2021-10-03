from typing import Tuple

import numpy as np
from scipy.io import wavfile as wf


def read_wav() -> Tuple[list, list]:
    """
    read the .wav file
    :return: samplerate, data
    """
    return wf.read('note_guitare_LAd.wav')


def fft(input_n: list) -> Tuple[list, list, list]:
    """
    fft the input
    :param input_n:
    :return: transformed input, module of output, phase of output
    """
    output_m = np.fft.fft(input_n)
    return output_m, np.abs(output_m), np.angle(output_m)
