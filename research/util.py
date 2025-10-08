"""
Utilities for loading/saving/playing sound
"""

from scipy.io import wavfile
import numpy as np
import IPython.display

global_rate = 48000


def wav_in(name: str):
    """
    read wav from ./in/
    """
    v = wavfile.read(f"./in/{name}.wav")
    global global_rate
    global_rate = v[0]
    return v


def wav_out(name: str, data: np.ndarray, rate: int | None = None):
    """
    write wav to ./out/
    data is 1D array or 2D array with shape (samples, channels)
    """
    if rate is None:
        rate = global_rate
    wavfile.write(f"./out/{name}.wav", rate, data)


def display_sound(data: np.ndarray, rate=None):
    """
    display a sound in the jupyter notebook
    """
    if rate is None:
        rate = global_rate
    return IPython.display.Audio(data=data.transpose(), rate=rate)
