import torch, random
import numpy as np
from functions import augmentation as augment_funcs
from functions import augmentation_param_search as augment_funcs_search

class DataLoader(object):

    def __init__(
        self, 
        dataset, 
        batchsize, 
        is_eval,
        augment=None, 
        device="cpu", 
        seed=1
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.seed = seed
        self.batchsize = batchsize

        self.shuffle = not is_eval # False if eval mode
        self.use_all = is_eval # Use all data if eval mode

        self.dataset = dataset
        self.augmentation = augment if augment is not None else "none"

        # Calculate number of batch based on batchsize
        div = len(self.dataset) // self.batchsize
        mod = len(self.dataset) % self.batchsize
        if is_eval:
            # if mod == 0 => num_batch = div
            # else => num_batch = div + 1 (include last)
            self.num_batch = div + int(mod > 0)
        else:
            # if div > 0 => num_batch = div
            # else => num_batch = 1 (only use last if batch size > dataset size)
            self.num_batch = div if div > 0 else 1

        self.device = device
        self.initialize()

    def initialize(self):
        if self.shuffle:
            self._shuffle_data()
        self.itercount = 0

    def _shuffle_data(self):
        idxs = np.arange(len(self.dataset))
        random.shuffle(idxs)

        self.dataset.data = self.dataset.data[idxs]
        self.dataset.label = self.dataset.label[idxs]

    def _numpy_to_torch(self, data):
        data = torch.from_numpy(data)
        try:
            data = data.to(self.device)
        except:
            pass
        return data

    def _augmentation(self, X_batch):
        if self.augmentation == "none":
            pass
        elif self.augmentation.startswith("rand"):
            X_batch = augment_funcs.randaug(X_batch, self.augmentation)
        elif self.augmentation.startswith("search"):
            X_batch = augment_funcs_search.fix_aug(X_batch, self.augmentation)
        elif self.augmentation.startswith("fix"):
            # fix-<Mval>-scale-drop...
            augmentation_info = self.augmentation.split("-")
            Mval = int(augmentation_info[1])
            for aug in augmentation_info[2:]:
                X_batch = augment_funcs.fix_aug(X_batch, aug, Mval)
        elif self.augmentation.startswith("exclude"):
            # exclude-<Mval>-XXX
            augmentation_info = self.augmentation.split("-")
            Mval = int(augmentation_info[1])
            aug = augmentation_info[2]
            assert(len(augmentation_info) == 3)
            X_batch = augment_funcs.exclude_aug(X_batch, aug, Mval)
        elif self.augmentation.startswith("all"):
            # all-<Mval>
            augmentation_info = self.augmentation.split("-")
            Mval = int(augmentation_info[1])
            X_batch = augment_funcs.all_aug(X_batch, Mval)
        else:
            raise NotImplementedError
            # for aug in self.augmentation.split("-"):
            #     X_batch = augment_funcs.fix_aug(X_batch, aug, 20)
        return X_batch

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        idx_s = self.itercount * self.batchsize
        idx_e = idx_s + self.batchsize

        if self.itercount == self.num_batch:
            self.initialize()
            raise StopIteration()
        else:
            self.itercount += 1

        X_batch, y_batch = self.dataset[idx_s:idx_e]
        X_batch = self._augmentation(X_batch)
        X_batch = self._numpy_to_torch(X_batch)
        y_batch = self._numpy_to_torch(y_batch)
        X_batch, y_batch = X_batch.float(), y_batch.long()
        return X_batch, y_batch

if __name__ == '__main__':
    pass
