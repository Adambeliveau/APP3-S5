import math

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from analysis import extract_sin, export_data
from filter import find_best_N, build_RIF
from plotting import add_plotting_args, build_plots
from symphony import build_symphony, synthesis
from utils import read_wav, write_wav, fft, convolution, normalisation

pp = PdfPages('figures.pdf')

if __name__ == '__main__':
    # extract sines meeting the specs
    f_e, x_n = read_wav('audio_files/note_guitare_LAd.wav')
    N_ech_guitar = len(x_n)
    w_lp = np.array([normalisation(m, N_ech_guitar) for m in range(N_ech_guitar)])
    w_lp_shifted = np.array([normalisation(m, N_ech_guitar, True) for m in range(N_ech_guitar)])
    window = np.hanning(N_ech_guitar)
    X_m_lp, mod_X_m_lp, phase_X_m_lp = fft(x_n * window)
    m, mod_X_m_lp, phase_X_m_lp = extract_sin(mod_X_m_lp, phase_X_m_lp, f_e, w_lp_shifted)

    # find the best N order for the low pass filter
    f_c = (math.pi / 1000 * f_e) / (2 * math.pi)
    N_low_pass = find_best_N(f_c, f_e, N_ech_guitar)

    # filter LA through low_pass to only get the tone
    f_c_lp = math.pi / 1000
    h_n_lp, mod_H_m_lp = build_RIF(N_low_pass, f_c_lp, f_e, 'low_pass', N_ech_guitar)

    # synthesise LA
    envelope = convolution(h_n_lp, np.abs(x_n))
    y_n_LA_sharp = synthesis(mod_X_m_lp, phase_X_m_lp, w_lp[m], envelope, 0)
    write_wav(f_e, y_n_LA_sharp.astype(np.int16), 'audio_files/LAd_filtered.wav')

    # make the 5th symphony out of the filtered LA from the guitar
    symphony = build_symphony(mod_X_m_lp, phase_X_m_lp, w_lp[m], envelope, f_e)
    write_wav(f_e, symphony.astype(np.int16), 'audio_files/symphony.wav')

    # filter 1000Hz sine in band stop
    f_e, x_n = read_wav('audio_files/note_basson_plus_sinus_1000_Hz.wav')
    N_ech_basson = len(x_n)
    N_band_stop = 6000
    f_c_bs = 1000
    h_n_bs, mod_H_m_bs = build_RIF(6000, f_c_bs, f_e, 'band_stop', N_ech_basson)
    w_bs = np.array([normalisation(m, N_ech_basson) for m in range(N_ech_basson)])
    w_bs_shifted = np.array([normalisation(m, N_ech_basson, True) for m in range(N_ech_basson)])

    # FFT to get the Fourier's spectrum
    X_m_bs, mod_X_m_bs, phase_X_m_bs = fft(x_n)

    # remove 1000Hz sine from signal and export it to a wav file
    y_n_basson = convolution(h_n_bs, x_n)
    # y_n = convolution(h_n_bs, y_n)
    # y_n = convolution(h_n_bs, y_n)
    write_wav(f_e, y_n_basson.astype(np.int16), 'audio_files/basson_filtered.wav')

    # Fourier's spectrum for filtered signals
    _, mod_Y_m_lp, phase_Y_m_lp = fft(y_n_LA_sharp)
    _, mod_Y_m_lp, phase_Y_m_lp = extract_sin(mod_Y_m_lp, phase_Y_m_lp, f_e, w_lp_shifted)
    _, mod_Y_m_bs, phase_Y_m_bs = fft(y_n_basson)

    w_bs_shifted_extended = np.array([normalisation(m, len(y_n_basson), True) for m in range(len(y_n_basson))])
    f_lp_shifted = np.array([(f_e * w) / (2 * math.pi) for w in w_lp_shifted])
    f_bs_shifted = np.array([(f_e * w) / (2 * math.pi) for w in w_bs_shifted])
    f_bs_shifted_extended = np.array([(f_e * w) / (2 * math.pi) for w in w_bs_shifted_extended])
    # plot parameters
    plotting_args = [
        # Fourier's spectrum
        add_plotting_args('stem', f_lp_shifted[m], 20 * np.log10(mod_X_m_lp),
                          'Module du spectre de Fourier du LA#', 'fréquence(Hz)', 'amplitude(dB)'),
        add_plotting_args('stem', f_lp_shifted[m], 20 * np.log10(mod_Y_m_lp),
                          'Module du spectre de Fourier du LA# filtré', 'fréquence(Hz)', 'amplitude(dB)'),
        add_plotting_args('plot', f_bs_shifted, 20 * np.log10(mod_X_m_bs),
                          'Module du spectre de Fourier du Basson', 'fréquence(Hz)', 'amplitude(dB)'),
        add_plotting_args('plot', f_bs_shifted_extended, 20 * np.log10(mod_Y_m_bs),
                          'Module du spectre de Fourier du Basson filtré', 'fréquence(Hz)', 'amplitude(dB)')]

    # plot building
    build_plots(pp, *plotting_args)

    export_data(f_lp_shifted[m], 20*np.log10(mod_X_m_lp), phase_X_m_lp)
