import os, pickle
import numpy as np
from functions import utils_data

from typing import Tuple, List
from nptyping import NDArray

class Dataset(object):

    def __init__(
        self, 
        data_dir, 
        datatype, 
        frequency, 
        seed, 
        max_duration,
        dataset, 
        pad_data=True, 
        num_class=9, 
        lead_type="I"
    ):

        assert(num_class in [3, 9])
        assert(lead_type in ["all", "I"])

        self.data_dir = os.path.expanduser(data_dir)
        data, label = self._load_labeled(datatype, seed)

        # Pad data if True
        if pad_data:
            max_length = max_duration * frequency
            data = utils_data.pad_batch(data, max_length, lead_type)
            self.data = data[:, np.newaxis, :]
        else:
            raise  NotImplementedError

        # Extract lead to use.
        if (dataset == "cpsc" and lead_type == "I"):
            self.data = self.data[:, :, 0]
        elif lead_type == "all":
            self.data = self.data[:, 0]

        # Process label
        if (dataset == "cpsc" and num_class == 3):
            label = self._process_label(label)
        self.label = np.array(label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def _load_labeled(self, datatype: str,
                      seed: int) -> Tuple[List, NDArray]:
        Xfile = f"{datatype}_X_seed{seed}.pkl"
        yfile = f"{datatype}_y_seed{seed}.pkl"
        X = self._open_pickle(Xfile)
        y = self._open_pickle(yfile)
        return X, y

    def _open_pickle(self, filename: str):
        file_loc = os.path.join(self.data_dir, filename)
        with open(file_loc, "rb") as fp:
            data = pickle.load(fp)
        return data

    def _process_label(self, label):
        # Convert CPSC classes into Physionet classes
        # CPSC label index 0 (Normal) => Physionet label 0
        # CPSC label index 1 (AF) => Physionet label 1
        # CPSC label index 2 - 8 (other classes) => Physionet label 2

        num_sample = len(label)
        new_label = np.zeros([num_sample]) - 1 # default to other class
        for i, l in enumerate(label):
            if l[0] == 1:
                new_label[i] = 0
            elif l[1] == 1:
                new_label[i] = 1
            else:
                new_label[i] = 2
        assert(min(new_label) > -1)
        return new_label

if __name__ == '__main__':
    pass
