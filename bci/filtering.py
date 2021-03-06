import os
import scipy.io as sio
import scipy.signal as sp
from os import listdir
from os.path import isfile, join
import numpy as np
import sklearn.linear_model as lm
import time
from scipy.signal import hilbert

# do stuff

Fs = 100  # set the samling rate
Nf = (Fs / 2)  # Nyquist frequency.


# How to choose order

def my_filtermy_filt(rate_1, rate_2, A):  # version according to Alex's code
    b_alpha, a_alpha = sp.iirfilter(4, Wn=np.array([rate_1 / (Fs / 2), rate_2 / (Fs / 2)]), btype='band',
                                    ftype='butter')
    alpha = sp.filtfilt(b_alpha, a_alpha, A)
    return alpha


def my_filter(Nf, rate_1,rate_2, A, order):
    b_alpha, a_alpha = sp.butter(order, Wn = np.array([rate_1/(Nf), rate_2/(Nf)]), btype='bandpass')
    alpha = y = sp.lfilter(b_alpha, a_alpha, A)

    print("my_filter was run")
    return alpha

def eneloper(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    print("eneloper was run")
    return amplitude_envelope


def three_envelope_of_signal(signal, order, Nf):
    theta = my_filter(Nf,4, 7, signal, order)
    alpha = my_filter(Nf,8, 15, signal, order)
    beta = my_filter(Nf,16, 31, signal, order)

    print("three_envelope_of_signal was run")
    return eneloper(theta), eneloper(alpha), eneloper(beta)