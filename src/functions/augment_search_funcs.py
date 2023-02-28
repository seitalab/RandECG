import numpy as np
from scipy import signal as scipy_signal
import biosppy

FREQ = 50
filter_order = 10

def search_random_scale(X, max_val, min_val):
    """
    rand (0 - 1) * (max_val - min_val) -> 0 ~ max_val - min_val
    -> + min_val -> min_val ~ max_val

    Args:

    Returns:

    """
    scaledX = X.copy()
    scaler = np.random.rand(X.shape[0]) * (max_val - min_val) + min_val
    scaledX *= scaler[:, np.newaxis, np.newaxis]
    return scaledX

def search_sine_noise(X, max_fs, min_fs):
    """
    Args:
        X:
        max_fs:
        min_fs:
    Returns:
        X_sine:
    """
    # Random Amplitude of 0 - 1
    Amp = np.random.rand()
    # Random frequency between min_fs - max_fs
    fs = np.random.rand() * (max_fs - min_fs) + min_fs

    duration = X.shape[2] / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * fs, X.shape[2])

    X_sine = X.copy()
    X_sine += Amp * np.sin(steps)
    return X_sine

def search_square_noise(X, max_fs, min_fs):
    """
    Args:
        X:
        max_fs (float):
        min_fs (float):
    Returns:
        X_square:
    """
    # Random Amplitude of 0 - 1
    Amp = np.random.rand()
    # Random frequency between min_fs - max_fs
    fs = np.random.rand() * (max_fs - min_fs) + min_fs

    duration = X.shape[2] / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * fs, X.shape[2])

    X_square = X.copy()
    X_square += Amp * scipy_signal.square(steps)
    return X_square

def search_sine_noise_partial(X, max_fs, min_fs):
    """
    Args:
        X:
        min_fs (float):
        max_fs (float):
    Returns:
        X_sine_p:
    """
    X_sine_p = X.copy()
    signal_length = X.shape[2]
    # Random frequency between min_fs - max_fs
    fs = np.random.rand() * (max_fs - min_fs) + min_fs

    duration = X.shape[2] / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * fs, signal_length)

    w_ratio = np.random.rand()
    width = int(signal_length * w_ratio)

    start = int(np.random.rand() * (signal_length - width))

    sine_curve = np.sin(steps)
    for w in range(width):
        X_sine_p[:, :, start+w] += sine_curve[np.newaxis, np.newaxis, w]
    return X_sine_p

def search_square_noise_partial(X, max_fs, min_fs):
    """
    Args:
        X:
        min_fs (float):
        max_fs (float):
    Returns:
        X_square_p:
    """
    X_square_p = X.copy()
    signal_length = X.shape[2]
    # Random frequency between min_fs - max_fs
    fs = np.random.rand() * (max_fs - min_fs) + min_fs

    duration = X.shape[2] / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * fs, signal_length)

    # Randomly choose width to add noise.
    w_ratio = np.random.rand()
    width = int(signal_length * w_ratio)

    start = int(np.random.rand() * (signal_length - width))

    square_pulse = scipy_signal.square(steps)
    for w in range(width):
        X_square_p[:, :, start+w] += square_pulse[np.newaxis, np.newaxis, w]
    return X_square_p

def search_shift_range(X, max_w, min_w):
    """
    Args:
        X:
        max_w:
        min_w:
    Returns:

    """
    X_shift = X.copy()
    signal_length = X.shape[2]
    width = np.random.rand() * (max_w - min_w) + min_w

    # random value between -1 to 1.
    randval = (np.random.rand() - 0.5 ) * 2
    shift = int(randval * signal_length * width)
    for i in range(len(X)):
        pad_size = np.zeros([X.shape[1], abs(shift)])
        if shift > 0:
            # shift direction: ->
            X_shift[i] = np.concatenate([pad_size, X[i, :, :-abs(shift)]],
                                        axis=1)
        elif shift < 0:
            # shift direction: <-
            X_shift[i] = np.concatenate([X[i, :, abs(shift):], pad_size],
                                        axis=1)
    return X_shift

def search_cutout_range(X, max_w, min_w):
    """
    Args:
        X :
        max_w:
        min_w:
    Returns:

    """
    Xcut = X.copy()
    signal_length = X.shape[2]
    width = np.random.rand() * (max_w - min_w) + min_w

    cut_width = int(signal_length * width)
    start = int(np.random.rand() * (signal_length - cut_width))
    for w in range(cut_width):
        Xcut[:, :, start+w] = 0
    return Xcut

def search_drop_range(X, max_r, min_r):
    """
    Args:
        X: np.array([batchsize, signal_length])
        max_w:
        min_w:
    Returns:
        Xdropped:
    """
    Xdropped = X.copy()
    seqlen = X.shape[-1]
    rate = np.random.rand() * (max_r - min_r) + min_r

    drop_loc = np.random.rand(seqlen) > rate
    Xdropped *= drop_loc[np.newaxis, np.newaxis, :]
    return Xdropped

def search_flip_range(X, max_r, min_r):
    """
    Flip signal if True.

    Args:
        X:
        max_w:
        min_w:
    Returns:
        Xflipped:
    """
    Xflipped = X.copy()
    rate = np.random.rand() * (max_r - min_r) + min_r
    flipper = np.random.rand(X.shape[0]) < rate
    # True -> 1 * (-2) -> -2 + 1 -> -1
    # False -> 0 * (-2) -> 0 + 1 -> 1
    flipper = flipper.astype("float") * -2 + 1
    Xflipped *= flipper[:, np.newaxis, np.newaxis]
    return Xflipped

def search_sine_partial_width(X, max_w, min_w):
    """
    Args:
        X:
        max_w:
        min_w:
    Returns:
        X_sine_p:
    """
    max_fs = 1.0
    min_fs = 0.1

    X_sine_p = X.copy()
    signal_length = X.shape[2]
    # Random frequency between min_fs - max_fs
    fs = np.random.rand() * (max_fs - min_fs) + min_fs

    duration = X.shape[2] / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * fs, signal_length)

    w_ratio = np.random.rand() * (max_w - min_w) + min_w
    width = int(signal_length * w_ratio)

    start = int(np.random.rand() * (signal_length - width))

    sine_curve = np.sin(steps)
    for w in range(width):
        X_sine_p[:, :, start+w] += sine_curve[np.newaxis, np.newaxis, w]
    return X_sine_p

def search_square_partial_width(X, max_w, min_w):
    """
    Args:
        X:
        max_w (float):
        min_w (float):
    Returns:
        X_square_p:
    """
    max_fs = 1.0
    min_fs = 0.02 #Fixed 200915

    X_square_p = X.copy()
    signal_length = X.shape[2]
    # Random frequency between min_fs - max_fs
    fs = np.random.rand() * (max_fs - min_fs) + min_fs

    duration = X.shape[2] / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * fs, signal_length)

    # Randomly choose width to add noise.
    w_ratio = np.random.rand() * (max_w - min_w) + min_w
    width = int(signal_length * w_ratio)

    start = int(np.random.rand() * (signal_length - width))

    square_pulse = scipy_signal.square(steps)
    for w in range(width):
        X_square_p[:, :, start+w] += square_pulse[np.newaxis, np.newaxis, w]
    return X_square_p

def search_wn_partial_width(X, max_w, min_w):
    """
    Args:
        X:
        max_w (float):
        min_w (float):
    Returns:
        X_wn_p:
    """

    X_wnp = X.copy()
    signal_length = X.shape[2]

    # Randomly choose width to add noise.
    w_ratio = np.random.rand() * (max_w - min_w) + min_w
    width = int(signal_length * w_ratio)

    start = int(np.random.rand() * (signal_length - width))

    white_noise = np.random.randn(X.shape[0], X.shape[1], width)
    for w in range(width):
        X_wnp[:, :, start+w] += white_noise[:, :, w]
    return X_wnp

def search_sine_amp(X, max_a, min_a):
    """
    Args:
        X:
        min_a:
        max_a:
    Returns:
        X_sine:
    """
    max_fs = 0.02
    min_fs = 0.001
    # Amplitude between min_a, max_a
    Amp = np.random.rand() * (max_a - min_a) + min_a
    # Random frequency between min_fs - max_fs
    fs = np.random.rand() * (max_fs - min_fs) + min_fs

    duration = X.shape[2] / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * fs, X.shape[2])

    X_sine = X.copy()
    X_sine += Amp * np.sin(steps)
    return X_sine

def search_square_amp(X, max_a, min_a):
    """
    Args:
        X:
        min_a (float):
        max_a (float):
    Returns:
        X_square:
    """
    max_fs = 0.1
    min_fs = 0.001
    # Amplitude between min_a, max_a
    Amp = np.random.rand() * (max_a - min_a) + min_a
    # Random frequency between min_fs - max_fs
    fs = np.random.rand() * (max_fs - min_fs) + min_fs

    duration = X.shape[2] / FREQ
    steps = np.linspace(0, 2 * np.pi * duration * fs, X.shape[2])

    X_square = X.copy()
    X_square += Amp * scipy_signal.square(steps)
    return X_square

def search_fir_l(X, max_fs, min_fs):
    """
    Args:

    Returns:

    """
    M = np.random.rand() * (max_fs - min_fs) + min_fs
    freq = min(int(M * FREQ), FREQ/2)
    filtered_X, _, _ = biosppy.tools.filter_signal(
        signal=X, ftype="FIR", band="lowpass", order=filter_order,
        frequency=freq, sampling_rate=FREQ)
    return filtered_X

def search_wn_amp(X, max_a, min_a):
    """
    Args:
        X:
        min_a (float):
        max_a (float):
    Returns:
        X_square:
    """

    # Amplitude between min_a, max_a
    Amp = np.random.rand() * (max_a - min_a) + min_a

    X_wn = X.copy()
    white_noise = np.random.randn(X.shape[0], X.shape[1], X.shape[2])
    X_wn += Amp * white_noise
    return X_wn
