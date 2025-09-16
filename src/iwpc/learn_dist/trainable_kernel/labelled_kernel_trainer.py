from pathlib import Path

from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.learn_dist.trainable_kernel.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.trainable_kernel.mixture_kernel import MixtureKernel
from iwpc.learn_dist.trainable_kernel.trainable_kernel_base import TrainableKernelBase
from iwpc.learn_dist.trainable_kernel.two_sided_exponential_kernel import TwoSidedExponentialKernel


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

    norm_kernel = GaussianKernel(
        3,
        0.,
        5e-7,
    )
    exp_kernel = TwoSidedExponentialKernel(
        3,
        0,
        5e-7,
        loc_model=norm_kernel.mean_model,
    )
    kernel = MixtureKernel([norm_kernel, exp_kernel], [0.9, 0.1])
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
