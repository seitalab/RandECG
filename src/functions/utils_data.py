
import numpy as np
from tqdm import tqdm
import config

# Padding related --------------------------------------------------

def pad_batch(X_batch, max_len, lead_type):
    # data: [[x, x, ..., x], ...] ([12, signal_length])
    # or data: [x, x, ..., x] ([signal_length])
    # pad + data: [0, 0, ..., 0, x, x, ..., x]

    lengths = np.array([ele.shape[-1] for ele in X_batch])

    pad_len = max_len - lengths
    Xnew = []
    for i in tqdm(range(lengths.size)):
        Xnew.append(add_blank(X_batch[i], pad_len[i]))
    Xdata = np.array(Xnew)
    return Xdata

def add_blank(data, padlen):
    # Pre pad
    if padlen == 0:
        return data
    elif padlen < 0:
        return data[:, -padlen:]

    if data.ndim == 1:
        pad = np.zeros([padlen])
    else:
        pad = np.zeros([data.shape[0], padlen])
    data = np.concatenate([pad, data], axis=-1)
    return data
