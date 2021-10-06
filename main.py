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
    f_e, x_n_LA = read_wav('audio_files/note_guitare_LAd.wav')
    N_ech_guitar = len(x_n_LA)
    w_lp = np.array([normalisation(m, N_ech_guitar) for m in range(N_ech_guitar)])
    w_lp_shifted = np.array([normalisation(m, N_ech_guitar, True) for m in range(N_ech_guitar)])
    window = np.hanning(N_ech_guitar)
    X_m_lp, mod_X_m_lp, phase_X_m_lp = fft(x_n_LA * window)
    f_fund_LAd = 466.2
    m, mod_X_m_lp, phase_X_m_lp = extract_sin(mod_X_m_lp, phase_X_m_lp, f_fund_LAd, f_e, w_lp_shifted)

    # find the best N order for the low pass filter
    f_c = (math.pi / 1000 * f_e) / (2 * math.pi)
    N_low_pass = find_best_N(f_c, f_e, N_ech_guitar)

    # filter LA through low_pass to only get the tone
    f_c_lp = math.pi / 1000
    h_n_lp, mod_H_m_lp, _ = build_RIF(N_low_pass, f_c_lp, f_e, 'low_pass', N_ech_guitar)

    # synthesise LA
    envelope_lp = convolution(h_n_lp, np.abs(x_n_LA))
    y_n_LA_sharp = synthesis(mod_X_m_lp, phase_X_m_lp, w_lp[m], envelope_lp, 0)
    envelope_lp_filtered = convolution(h_n_lp, np.abs(y_n_LA_sharp))
    write_wav(f_e, y_n_LA_sharp.astype(np.int16), 'audio_files/LAd_synthesized.wav')

    # make the 5th symphony out of the filtered LA from the guitar
    symphony = build_symphony(mod_X_m_lp, phase_X_m_lp, w_lp[m], envelope_lp, f_e)
    write_wav(f_e, symphony.astype(np.int16), 'audio_files/symphony.wav')

    # filter 1000Hz sine in band stop
    f_e, x_n_basson = read_wav('audio_files/note_basson_plus_sinus_1000_Hz.wav')
    N_ech_basson = len(x_n_basson)
    N_band_stop = 6000
    f_c_bs = 1000
    h_n_bs, mod_H_m_bs, phase_H_m_bs = build_RIF(6000, f_c_bs, f_e, 'band_stop', N_ech_basson)
    w_bs = np.array([normalisation(m, N_ech_basson) for m in range(N_ech_basson)])
    w_bs_shifted = np.array([normalisation(m, N_ech_basson, True) for m in range(N_ech_basson)])

    # FFT to get the Fourier's spectrum
    X_m_bs, mod_X_m_bs, phase_X_m_bs = fft(x_n_basson)

    # remove 1000Hz sine from signal and export it to a wav file
    envelope_bs = convolution(h_n_lp, np.abs(x_n_basson))
    y_n_basson = convolution(h_n_bs, x_n_basson)
    y_n_basson = convolution(h_n_bs, y_n_basson)
    y_n_basson = convolution(h_n_bs, y_n_basson)
    envelope_bs_filtered = convolution(h_n_lp, np.abs(y_n_basson))

    # synthesis basson
    f_fund_basson = 246.94
    m_basson, mod_X_m_bs, phase_X_m_bs = extract_sin(mod_X_m_bs, phase_X_m_bs, f_fund_basson,  f_e, w_bs_shifted)
    basson_synthesized = synthesis(mod_X_m_bs, phase_X_m_bs, w_bs[m_basson], envelope_bs_filtered, 0)
    write_wav(f_e, basson_synthesized.astype(np.int16), 'audio_files/basson_synthesised.wav')

    # Fourier's spectrum for filtered signals
    _, mod_Y_m_lp, phase_Y_m_lp = fft(y_n_LA_sharp)
    m, mod_Y_m_lp, phase_Y_m_lp = extract_sin(mod_Y_m_lp, phase_Y_m_lp, f_fund_LAd, f_e, w_lp_shifted)
    _, mod_Y_m_bs, phase_Y_m_bs = fft(basson_synthesized)
    m_basson, mod_Y_m_bs, phase_Y_m_bs = extract_sin(mod_Y_m_bs, phase_Y_m_bs, f_fund_basson,  f_e, w_bs_shifted)

    f_lp_shifted = np.array([(f_e * w) / (2 * math.pi) for w in w_lp_shifted])
    f_bs_shifted = np.array([(f_e * w) / (2 * math.pi) for w in w_bs_shifted])

    # 1000Hz sine vs band-stop filter
    sine = np.sin([2 * np.pi * 1000 * n / f_e for n in np.arange(len(x_n_basson))])
    y_n_sine = convolution(h_n_bs, sine)
    y_n_sine = convolution(h_n_bs, y_n_sine)
    y_n_sine = convolution(h_n_bs, y_n_sine)
    w_bs_shifted_extended = np.array([normalisation(m, len(y_n_basson), True) for m in range(len(y_n_sine))])
    f_bs_shifted_extended = np.array([(f_e * w) / (2 * math.pi) for w in w_bs_shifted_extended])

    # plot parameters
    plotting_args = [
        # Fourier's spectrum
        add_plotting_args('stem', f_lp_shifted[m], 20 * np.log10(mod_X_m_lp),
                          'Modules du spectre de Fourier du LA#', 'fréquence(Hz)', 'amplitude(dB)'),
        add_plotting_args('stem', f_lp_shifted[m], 20 * np.log10(mod_Y_m_lp),
                          'Modules du spectre de Fourier de la synthèse du LA#', 'fréquence(Hz)', 'amplitude(dB)'),
        add_plotting_args('stem', f_bs_shifted[m_basson], 20 * np.log10(mod_X_m_bs),
                          'Modules du spectre de Fourier du Basson sans la sinusoide de 1000Hz', 'fréquence(Hz)', 'amplitude(dB)'),
        add_plotting_args('stem', f_bs_shifted[m_basson], 20 * np.log10(mod_Y_m_bs),
                          'Modules du spectre de Fourier de la synthèse du Basson sans la sinusoide de 1000Hz', 'fréquence(Hz)', 'amplitude(dB)'),
        add_plotting_args('plot', None, envelope_lp,
                          'Enveloppe temporelle du LA#', 'n(échantillons)', 'amplitude',),
        add_plotting_args('plot', None, envelope_bs_filtered,
                          'Enveloppe temporelle du basson sans la sinusoide de 1000Hz', 'n(échantillons)', 'amplitude'),
        add_plotting_args('plot', w_lp, 20*np.log10(mod_H_m_lp),
                          'Réponse fréquentielle H[w] du filtre passe-bas', 'w(rad/échantillons)', 'amplitude(dB)'),
        add_plotting_args('plot', None, h_n_bs,
                          'Réponse impulsionnelle h[n] du filtre coupe-bande', 'n(échantillons)', 'amplitude'),
        add_plotting_args('plot', w_bs, 20*np.log10(mod_H_m_bs),
                          'Modules de la réponse fréquentielle H[w] du filtre coupe-bande', 'w(rad/échantillons)', 'amplitude(dB)'),
        add_plotting_args('plot', w_bs, phase_H_m_bs,
                          'Phases de la réponse fréquentielle H[w] du filtre coupe-bande', 'w(rad/échantillons)', 'phases(rad)'),
        add_plotting_args('plot', None, x_n_basson, "Spectres d'amplitude du signal du basson", 'n(échantillons)', 'amplitude'),
        add_plotting_args('plot', None, y_n_basson, "Spectres d'amplitude du signal du basson filtré", 'n(échantillons)', 'amplitude'),
        add_plotting_args('plot', f_bs_shifted_extended, y_n_sine,
                          "Réponse d'une sinusoïde de 1000Hz", 'fréquence(Hz)', 'amplitude')
    ]

    # plot building
    build_plots(pp, *plotting_args)

    export_data(f_lp_shifted[m], 20*np.log10(mod_X_m_lp), phase_X_m_lp, f_bs_shifted[m_basson], 20*np.log10(mod_X_m_bs), phase_X_m_bs)