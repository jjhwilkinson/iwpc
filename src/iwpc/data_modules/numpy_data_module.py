from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from numpy._typing import NDArray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class BinaryNumpyDataModule(LightningDataModule):
    """
    A DataModule which wraps a pair of numpy arrays containing the features associated with samples from two different
    classes and their weights
    """
    def __init__(
        self,
        p_samples: NDArray,
        q_samples: NDArray,
        p_weights: Optional[NDArray] = None,
        q_weights: Optional[NDArray] = None,
        validation_split: Optional[float] = 0.5,
        dataloader_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        p_samples
            A numpy array of shape (num_p_samples, num_features) containing the features associated with samples from
            the first category
        q_samples
            A numpy array of shape (num_q_samples, num_features) containing the features associated with samples from
            the second category. Must have the same number of features as p_samples, but need not contain the same
            number of samples
        p_weights
            A numpy array of shape (num_p_samples,) containing the weights associated with samples from the
            first category
        q_weights
            A numpy array of shape (num_q_samples,) containing the weights associated with samples from the
            second category
        validation_split
            The train-validation split to use. Must be between 0 and 1 and represents the fraction of data used for
            training. Defaults to 0.5
        dataloader_kwargs
            Any other arguments to be provided to DataLoader instances
        """
        super().__init__()
        self.p_samples = p_samples
        self.q_samples = q_samples
        self.p_weights = (np.ones(p_samples.shape[0]) if p_weights is None else p_weights).copy()
        self.p_weights /= self.p_weights.mean()
        self.q_weights = (np.ones(q_samples.shape[0]) if q_weights is None else q_weights).copy()
        self.q_weights /= self.q_weights.mean()
        self.validation_split = validation_split
        self.dataloader_kwargs = dataloader_kwargs
        self.ndim = p_samples.shape[1]

        all_samples = torch.as_tensor(np.concatenate((self.p_samples, self.q_samples)), dtype=torch.float32)
        all_weights = torch.as_tensor(np.concatenate((self.p_weights, self.q_weights)), dtype=torch.float32)
        all_labels = torch.as_tensor(
            np.concatenate((np.zeros(self.p_samples.shape[0]), np.ones(self.q_samples.shape[0]))),
            dtype=torch.float32,
        )
        self.all_data_ds = TensorDataset(all_samples, all_labels, all_weights)
        self.train_ds, self.val_ds = train_test_split(
            self.all_data_ds,
            train_size=self.validation_split,
            shuffle=True,
        )

    def train_dataloader(self):
        """
        Returns
        -------
        DataLoader
            A DataLoader instance initialised with the training portion of the original data
        """
        dataloader_kwargs = self.dataloader_kwargs.copy()
        dataloader_kwargs.setdefault("shuffle", True)
        return DataLoader(
            self.train_ds,
            **dataloader_kwargs
        )

    def val_dataloader(self):
        """
        Returns
        -------
        DataLoader
            A DataLoader instance initialised with the validation portion of the original data
        """
        return DataLoader(
            self.val_ds,
            **self.dataloader_kwargs
        )
