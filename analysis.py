import math
from typing import Tuple

import numpy as np
import pandas as pd

from utils import normalisation


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


def export_data(frequencies: np.ndarray, mod_X_m: np.ndarray, phase_X_m: np.ndarray) -> None:
    """
    Export frequency, module and phase from the selected harmonics to a csv file
    :param frequencies: frequencies of the selected harmonics
    :param mod_X_m: modules of the selected harmonics
    :param phase_X_m: phases of the selected harmonics
    :return: None
    """
    df = pd.DataFrame({'Fréquences': frequencies, 'Modules': mod_X_m, 'Phases': phase_X_m})
    df.to_csv('CSV_files/exported_data.csv')
