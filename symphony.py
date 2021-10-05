import numpy as np
from tqdm import tqdm


def synthesis(mod_X_m: np.ndarray, phase_X_m: np.ndarray, w: np.ndarray,
              envelope: np.ndarray, k: int) -> np.ndarray:
    """
    add sin and multiply the result with the envelope
    :param mod_X_m: modules of the signal
    :param phase_X_m: phases of the signal
    :param w: omega[m]
    :param envelope: envelope from the convolution
    :param k: k index for different notes
    :return:
    """
    f = (2**(k/12))
    summation = np.zeros(len(envelope))
    for n in tqdm(range(len(envelope)), desc='computing the 5th symphony'):
        summation[n] = np.sum(np.multiply(mod_X_m/max(mod_X_m), np.sin(np.multiply(f*n, w) + phase_X_m)))

    return np.multiply(summation, envelope)


def build_symphony(mod_X_m: np.ndarray, phase_X_m: np.ndarray, w_m: np.ndarray, envelope: np.ndarray, f_e: float) -> np.ndarray:
    """
    Builds the 5th symphony from Beetoven from the guitar note
    :param mod_X_m: module X[m]
    :param phase_X_m: phase X[m]
    :param w_m: omega w[m]
    :param envelope: envelope obtained from the convolution(h_n, |x[n]|)
    :param f_e: sample rate
    :return: symphony
    """
    symphony = np.array([])
    for val in [(3, -3), (1, -7), 'silence', (3, -5), (1, -8)]:
        if val == 'silence':
            symphony = np.append(symphony, np.zeros(int(f_e / 3)))
        else:
            nb_note, k = val
            nb_sample = f_e if k in [-6, -7] else int(f_e / 4)
            for i in range(nb_note):
                note = (synthesis(mod_X_m, phase_X_m, w_m, envelope, k))[nb_sample:2 * nb_sample]
                symphony = np.append(symphony, note)

    return symphony
