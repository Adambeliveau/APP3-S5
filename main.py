import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.io import wavfile as wf
import matplotlib.pyplot as plt


def read_wav():
    return wf.read('note_guitare_LAd.wav')


def fft(x_n):
    X_m = np.fft.fft(x_n)
    return X_m, np.abs(X_m), np.angle(X_m)


def extract_good_sin(mod_X_m, deg_X_m):
    # requirements: 10% max amplitude, amplitude[m] > amplitude[m+1] et amplitude[m] > amplitude[m-1]

    max_amplitude = max(mod_X_m)
    req_max_amplitude = 0.1 * max_amplitude

    good_mod_X_m = []
    good_deg_X_m = []
    m_list = []
    for i, amplitude in enumerate(mod_X_m):
        if amplitude >= req_max_amplitude:
            if i != len(mod_X_m) - 1:
                if mod_X_m[i - 1] < amplitude > mod_X_m[i + 1]:
                    good_deg_X_m.append(deg_X_m[i])
                    good_mod_X_m.append(amplitude)
                    m_list.append([i])
    return m_list, good_mod_X_m, good_deg_X_m


def plot_wav():
    f_e, x_n = read_wav()
    X_m, mod_X_m, deg_X_m = fft(x_n)

    m, mod_X_m, deg_X_m = extract_good_sin(mod_X_m, deg_X_m)

    fig, ax = plt.subplots(ncols=1, nrows=3)

    ax[0].plot(X_m)
    ax[0].set_title('X_m')
    ax[1].stem(m, mod_X_m)
    ax[1].set_title('module X_m')
    ax[2].stem(m, deg_X_m)
    ax[2].set_title('phase X_m')

    plt.tight_layout(pad=3.0)
    plt.show()


plot_wav()