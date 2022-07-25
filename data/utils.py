from torch.utils.data import Dataset


class DatasetJoiner(Dataset):

    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2
        self.len_ds1 = len(self.ds1)

    def __len__(self):
        return len(self.ds1) + len(self.ds2)

    def __getitem__(self, idx):
        if idx >= self.len_ds1:
            idx -= self.len_ds1
            return self.ds2[idx]
        else:
            return self.ds1[idx]
