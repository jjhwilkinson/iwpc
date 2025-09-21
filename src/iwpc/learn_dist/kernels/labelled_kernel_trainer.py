from pathlib import Path

import numpy as np
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.encodings.continuous_periodic_encoding import ContinuousPeriodicEncoding
from iwpc.encodings.discrete_log_prob_encoding import DiscreteLogProbEncoding
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.mixture_kernel import MixtureKernel
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.learn_dist.kernels.two_sided_exponential_kernel import TwoSidedExponentialKernel
from iwpc.models.layers import ConstantScaleLayer
from iwpc.models.utils import basic_model_factory


class LabelledKernelTrainer(LightningModule):
    def __init__(self, kernel: TrainableKernelBase):
        super().__init__()
        self.kernel = kernel

    def calculate_loss(self, batch):
        cond, targets, _ = batch
        log_prob = self.kernel.log_prob(targets, cond)
        return - log_prob.mean()

    def training_step(self, batch):
        loss = self.calculate_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        loss = self.calculate_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=10,
                    factor=0.1,
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


if __name__ == '__main__':
    dm = PandasDirDataModule(
        # Path("/Users/jeremywilkinson/research_data/Thesis/learn_det_resp/test_data"),
        Path("/Users/jeremywilkinson/research_data/Thesis/learn_det_resp/user.jjwilkin.SymmetryAnalysis.mc23d.Zmumu.MuonDetResp.2506091315.root_ANALYSIS_prepped_flat"),
        ['q_over_pt', 'theta', 'phi'],
        target_cols=['q_over_pt_err'],
        dataloader_kwargs={'num_workers': 5},
    )

    cond_encoding = TrivialEncoding(2) & ContinuousPeriodicEncoding()
    loc_model = basic_model_factory(
        cond_encoding,
        1,
        final_layers=[ConstantScaleLayer(scale=5e-7)]
    )
    std_model = basic_model_factory(
        cond_encoding,
        ExponentialEncoding(1),
        final_layers=[ConstantScaleLayer(shift=np.log(5e-7)),]
    )
    scale_model = basic_model_factory(
        cond_encoding,
        ExponentialEncoding(1),
        final_layers=[ConstantScaleLayer(shift=np.log(3 * 5e-7)),]
    )
    log_weight_model = basic_model_factory(
        cond_encoding,
        DiscreteLogProbEncoding(2),
        final_layers=[ConstantScaleLayer(shift=np.log([0.9, 0.1]))]
    )

    norm_kernel = GaussianKernel(
        cond_encoding,
        loc_model,
        std_model,
    )
    exp_kernel = TwoSidedExponentialKernel(
        cond_encoding,
        loc_model,
        scale_model,
    )
    kernel = MixtureKernel(
        cond_encoding,
        [norm_kernel, exp_kernel],
        log_weight_model,
    )
    module = LabelledKernelTrainer(kernel)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
    )
    trainer = Trainer(
        callbacks=[LearningRateMonitor(logging_interval='epoch'), checkpoint_callback],
        accelerator="mps",
        num_sanity_val_steps=0,
    )

    with dm.tmp_transform(lambda df: df.loc[df['is_reco_matched'] == 1]) as dm:
        trainer.fit(
            module,
            datamodule=dm,
        )
