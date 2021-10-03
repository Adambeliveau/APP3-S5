import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile as wf
from scipy import signal
from tqdm import tqdm


def read_wav():
    return wf.read('note_guitare_LAd.wav')


def fft(x_n):
    X_m = np.fft.fft(x_n)
    return X_m, np.abs(X_m), np.angle(X_m)


def extract_sin(mod_X_m, deg_X_m):
    # requirements: 10% max amplitude, amplitude[m] > amplitude[m+1] et amplitude[m] > amplitude[m-1]

    max_amplitude = max(mod_X_m)
    req_max_amplitude = 0.1 * max_amplitude

    extracted_mod_X_m = []
    extracted_deg_X_m = []
    extracted_mod_X_m_dB = []
    m_list = []
    for i, amplitude in enumerate(mod_X_m):
        if amplitude >= req_max_amplitude:
            if i != len(mod_X_m) - 1:
                if mod_X_m[i - 1] < amplitude > mod_X_m[i + 1]:
                    extracted_deg_X_m.append(deg_X_m[i])
                    extracted_mod_X_m.append(amplitude)
                    extracted_mod_X_m_dB.append(20 * math.log(amplitude))
                    m_list.append(i)
    return m_list, extracted_mod_X_m, extracted_mod_X_m_dB, extracted_deg_X_m


def build_RIF(N):
    m = int((math.pi / 1000) * N / (2 * math.pi))
    k = 2 * m + 1
    h_n = [(1 / N) * (math.sin(math.pi * n * k / N) / math.sin(math.pi * n / N)) for n in range(int(-N/2)+1, int(N/2)) if n != 0]
    h_n.insert(math.ceil(len(h_n)/2), k / N)
    H_m = reponse_freq(h_n)
    return h_n, np.abs(H_m)


def reponse_freq(h_n):
    return np.fft.fft(np.concatenate((h_n, np.zeros(160000-len(h_n)))))


def plot_wav():
    f_e, x_n = read_wav()
    X_m, mod_X_m, deg_X_m = fft(x_n)

    m, mod_X_m, mod_X_m_dB, deg_X_m = extract_sin(mod_X_m, deg_X_m)

    # fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(10, 10))
    # ax[0].plot(X_m)
    # ax[0].set_title('X_m')
    # ax[1].stem(m, mod_X_m)
    # ax[1].set_title('module X_m')
    # ax[1].set_ylabel('amplitude')
    # ax[1].set_xlabel('m')
    # ax[2].stem(m, mod_X_m_dB)
    # ax[2].set_title('module X_m')
    # ax[2].set_ylabel('amplitude(dB)')
    # ax[2].set_xlabel('m')
    # ax[3].stem(m, deg_X_m)
    # ax[3].set_title('phase X_m')
    # ax[3].set_ylabel('phase(deg)')
    # ax[3].set_xlabel('m')
    # plt.tight_layout(pad=3.0)

    range_start = 800
    range_stop = 900
    range_step = 1
    deltas = []
    for N in tqdm(range(range_start, range_stop, range_step)):
        h_n, H_m = build_RIF(N)
        mod_H_m = 20*np.log10(H_m)
        w = [2 * math.pi * m / len(H_m) for m in range(len(H_m))]
        deltas.append(-3 - np.interp(math.pi / 1000, w, mod_H_m))

    print(f'closest delta: {deltas[(np.abs(deltas)).argmin()]} | corresponding N: {range_start + (np.abs(deltas)).argmin()}')

    # plt.legend(np.arange(range_start, range_stop, range_step).astype(str))
    # plt.show()


if __name__ == '__main__':
    plot_wav()
