import math

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from plotting import add_plotting_args, build_plots
from filter import find_best_N, build_RIF
from symphony import build_symphony, synthesis
from utils import read_wav, write_wav, extract_sin, fft, convolution, normalisation

pp = PdfPages('figures.pdf')

if __name__ == '__main__':
    # extract sines meeting the specs
    f_e, x_n = read_wav('audio_files/note_guitare_LAd.wav')
    N_ech = len(x_n)
    w_lp = np.array([normalisation(m, len(x_n)) for m in range(len(x_n))])
    w_lp_shifted = np.array([normalisation(m, len(x_n), True) for m in range(len(x_n))])
    window = np.hanning(len(x_n))
    X_m, mod_X_m, phase_X_m = fft(x_n * window)
    m, mod_X_m, phase_X_m = extract_sin(mod_X_m, phase_X_m, f_e, w_lp_shifted)

    # find the best N order for the low pass filter
    f_c = (math.pi/1000*f_e)/(2 * math.pi)
    N_low_pass = find_best_N(f_c, f_e, N_ech)

    # filter LA through low_pass to only get the tone
    f_c_lp = math.pi/1000
    h_n_lp, mod_H_m_lp = build_RIF(N_low_pass, f_c_lp, f_e, 'low_pass', N_ech)

    # synthesise LA
    envelop = convolution(h_n_lp, np.abs(x_n))
    LA = synthesis(mod_X_m, phase_X_m, w_lp[m], envelop, 0)
    write_wav(f_e, LA.astype(np.int16), 'audio_files/LA_filtered.wav')

    # make the 5th symphony out of the filtered LA from the guitar
    symphony = build_symphony(mod_X_m, phase_X_m, w_lp[m], envelop, f_e)
    write_wav(f_e, symphony.astype(np.int16), 'audio_files/symphony.wav')

    # filter 1000Hz sine in band stop
    f_e, x_n = read_wav('audio_files/note_basson_plus_sinus_1000_Hz.wav')
    N_band_stop = 6000
    f_c_bs = 1000
    h_n_bs, mod_H_m_bs = build_RIF(6000, f_c_bs, f_e, 'band_stop', len(x_n))
    w_bs = np.array([normalisation(m, len(mod_H_m_bs)) for m in range(len(mod_H_m_bs))])

    # normalise with the ration m/N than use the result to extract f from w = 2*pi*f/f_e
    f = np.array([(normalisation(m, len(x_n))*f_e)/(2 * math.pi) for m in range(len(x_n))])

    # remove 1000Hz sine from signal and export it to a wav file
    y_n = convolution(h_n_bs, x_n)
    write_wav(f_e, y_n.astype(np.int16), 'audio_files/basson_filtered.wav')

    # plot parameters
    plotting_args = [add_plotting_args('plot', None, X_m, 'FFT du signal x[n] de la guitare', 'n(échantillons}', 'amplitude'),
                     add_plotting_args('stem', w_lp_shifted[m], 20 * np.log10(mod_X_m), '|X[w]| de la guitare', 'w(rad/échantillons)', 'amplitude(dB)'),
                     add_plotting_args('stem', w_lp_shifted[m], phase_X_m, 'phase X[w] de la guitare', 'w(rad/échantillons)', 'phase(rad)'),
                     add_plotting_args('stem', range(int(-N_low_pass / 2) + 1, int(N_low_pass / 2)), h_n_lp, f'réponse impulsionnelle du filtre passe-bas N={N_low_pass}', 'm(échatillons)', 'amplitude'),
                     add_plotting_args('plot', w_lp, 20*np.log10(mod_H_m_lp), f'réponse fréquencielle du filtre passe-bas N={N_low_pass}', 'w(rad/échantillons)', 'amplitude(dB)'),
                     add_plotting_args('plot', None, LA, 'note LA filtré à traver le passe-bas de N=884', 'n(échantillons)', 'amplitude'),
                     add_plotting_args('plot', None, symphony, '5e symphony de Beethoven', 'n(échantillons)', 'amplitude'),
                     add_plotting_args('plot', None, h_n_bs, f'réponse impulsionnelle du filtre passe-bas N={N_band_stop}', 'm(échatillons)', 'amplitude'),
                     add_plotting_args('plot', w_bs, 20 * np.log10(mod_H_m_bs), f'réponse fréquencielle du filtre passe-bas N={N_band_stop}', 'm(échantillons)', 'amplitude(dB)'),
                     add_plotting_args('plot', None, y_n, 'signal filtré du basson y[n]', 'n(échantillons)', 'amplitude')]

    # plot building
    build_plots(pp, *plotting_args)
