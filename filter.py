#from Package_before_deployment.Preprocess.filter import sig_filter
import math
import numpy as np
import pandas as pd
import scipy as sp
import scipy.fftpack



def sig_filter(sig, cutoff, Butter_filter=False, ds_fsctor = 1 ):
    """
    Performs lowpass filtering on the one-dimensional input signal
    Args: 
    sig: one dimensional signal to be filtered
    cutoff (Hz): low pass filter cutoff  frequncy
    Butter_filter (bool): If true butter worth filter is computed
    df_factor: the decimation factor(the default is 1)
    Returns:
    lowpass filtered one dimensional array
    """
    if Butter_filter == True:
        filtered = butter_lowpass_filter(sig,cutoff,fs=100,order=2)[::ds_fsctor]
    else:
        sig_=sig.values.copy()

        temp_fft = sp.fftpack.fft(sig_)
        temp_fft_bis = temp_fft.copy()
        fftfreq = sp.fftpack.fftfreq(len(temp_fft_bis), 1. / 100)
        temp_fft_bis[np.abs(fftfreq) > cutoff] = 0
        filtered = np.real(sp.fftpack.ifft(temp_fft_bis))
    return filtered   