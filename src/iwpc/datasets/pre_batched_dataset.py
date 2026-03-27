import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class PreBatchedDataset(Dataset):
    @staticmethod
    def collator_fn(batch):
        return batch[0]

    def __init__(self, tensors, device, batch_size=2**17):
        super().__init__()
        self.device = device
        self.batch_size = int(batch_size)

        tensors = [torch.as_tensor(t) for t in tensors]

        self.batches = []
        for i in range(0, tensors[0].shape[0], batch_size):
            self.batches.append(tuple(torch.as_tensor(t[i:i+batch_size], device=device, dtype=torch.float32) for t in tensors))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


if __name__ == '__main__':
    df = pd.read_pickle("/Users/jeremywilkinson/research_data/Thesis/kernel/user.jjwilkin.SymmetryAnalysis.mc23d.Zmumu.MuonDetResp.2506091315.root_ANALYSIS_prepped/file_0.pkl")
    ds = PreBatchedDataset([df[['cond_mu_plus_q_over_pt', 'cond_mu_minus_q_over_pt']].values, df[['label']].values, np.ones(df.shape[0])], 'mps')
    dl = DataLoader(ds, batch_size=1, collate_fn=PreBatchedDataset.collator_fn)
    # print(len(ds[0]))
    print(next(iter(dl)))