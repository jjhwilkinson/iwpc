# `iwpc.data_modules`

PyTorch Lightning `DataModule` wrappers that feed the standard `iwpc` batch
contract — `(features, labels, weights)` with `labels == 0` for samples drawn
from `p` and `labels == 1` for samples drawn from `q` — into
`calculate_divergence`, `run_reweight_loop`, and the rest of the training
machinery.

## Purpose

Every estimator in `iwpc` consumes `(features, labels, weights)` batches. This
sub-package is the small set of `LightningDataModule` adapters that produce
those batches from the three input formats the codebase actually uses:

- a single in-memory pandas `DataFrame` (or a `(p_df, q_df)` pair),
- a pair of numpy arrays for `p` and `q`,
- a directory of pickled DataFrame shards on disk (the only format that the
  reweight loop and the long-running examples use).

## Layout

| Module | Class(es) | When to use |
| --- | --- | --- |
| `pandas_data_module.py` | `PandasDataModule`, `BinaryPandasDataModule` | Data already lives in a `DataFrame` (or two) in memory |
| `numpy_data_module.py` | `BinaryNumpyDataModule` | `p` and `q` are plain numpy arrays |
| `pandas_directory_data_module.py` | `PandasDirDataModule` | Dataset is too large to hold in memory, or you need `.transform` / `.reweight` / `.copy` / tags (required by `run_reweight_loop`) |
| `pandas_directory_data_module_builder.py` | `PandasDirDataModuleBuilder` | Context manager for assembling a new on-disk dataset shard-by-shard |

All four data modules split 50/50 train/validation by default
(`validation_split=0.5` or `split=0.5`). The pandas/numpy modules call
`sklearn.train_test_split` over the concatenated rows; `PandasDirDataModule`
splits *by file* — see below.

## Usage

### `BinaryPandasDataModule` from a `(p_df, q_df)` tuple

`p_df` and `q_df` are concatenated and a `__label` column is added
automatically (`0` for `p_df`, `1` for `q_df`).

```python
from iwpc.data_modules.pandas_data_module import BinaryPandasDataModule

dm = BinaryPandasDataModule(
    p_df=p_df,
    q_df=q_df,
    feature_cols=["x", "y"],
    weight_col="w",                       # optional
    validation_split=0.5,
    dataloader_kwargs={"batch_size": 2**15, "num_workers": 4},
)
```

The resulting data module yields `(features, __label, weight)` triples ready
for `calculate_divergence`.

### `BinaryNumpyDataModule` from two ndarrays

```python
import numpy as np
from iwpc.data_modules.numpy_data_module import BinaryNumpyDataModule

p_samples = np.random.randn(100_000, 3)
q_samples = np.random.randn(120_000, 3) + 0.3

dm = BinaryNumpyDataModule(
    p_samples=p_samples,
    q_samples=q_samples,
    p_weights=None,            # optional; defaults to 1s, then mean-normalised
    q_weights=None,
    validation_split=0.5,
    dataloader_kwargs={"batch_size": 2**15},
)
```

Labels (`0` for `p`, `1` for `q`) are added internally and the weights are
mean-normalised inside each class.

### `PandasDirDataModule` pointing at a sharded directory

The directory must contain `file_0<ext> … file_{N-1}<ext>` and a `ds_info.yml`
listing `file_sizes` (in file order). `<ext>` is the serializer's extension —
`.pkl` by default, `.parquet` when built with the parquet serializer:

```
sample_dataset/
  ds_info.yml          # contains: file_sizes: [100000, 100000]
  file_0.pkl
  file_1.pkl
```

#### Serialization format

The on-disk shard format is pluggable via the `serializer` argument on both
`PandasDirDataModule` and `PandasDirDataModuleBuilder`. It accepts a
`DataFrameSerializer` instance, the name of a built-in (`"pickle"` — the
default — or `"parquet"`), or `None` to auto-detect. Pickle stays the default
so existing datasets and configs are unchanged. Parquet (via `pyarrow`) is
numpy-version-neutral and avoids the `ModuleNotFoundError: No module named
'numpy._core.numeric'` you get loading a numpy-2 pickle under numpy 1.x. The
chosen format is recorded in `ds_info.yml` under a `serializer` field, so a
dataset re-opens with the right reader without re-specifying it (a `ds_info.yml`
with no `serializer` field defaults to pickle):

```python
PandasDirDataModuleBuilder("new_dataset", serializer="parquet")  # writes file_i.parquet
PandasDirDataModule("new_dataset")                               # auto-detects parquet
```

```python
from pathlib import Path
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule

dm = PandasDirDataModule(
    Path("sample_dataset"),
    feature_spec=["x", "y"],          # passed straight to PandasDataset
    weight_col="w",                   # optional
    split=0.5,                        # first ceil(N*split) files = train
    use_in_memory_dataset=False,      # set True to preload into shared memory
)
```

Train/val split is *by file*: the first `ceil(N * split)` files go to
training, the remainder to validation. **There is no shuffling across the
file boundary**, so on-disk file ordering must already be unbiased — use
`dm.shuffle()` or `PandasDirDataModuleBuilder(..., shuffle=True)` if you are
not sure.

#### Building a new sharded dataset

```python
from iwpc.data_modules.pandas_directory_data_module_builder import (
    PandasDirDataModuleBuilder,
)

with PandasDirDataModuleBuilder(
    "new_dataset",
    file_size=5_000_000,
    shuffle=True,
    tags=["source: experiment_42"],
) as builder:
    for df in some_iterator:
        builder.write(df)
```

The builder writes `file_i<ext>` shards (`.pkl` by default; pass
`serializer="parquet"` to write `.parquet`), records each `file_sizes` entry,
dumps `ds_info.yml` (with a creation timestamp tag prepended and the chosen
`serializer` recorded), and optionally re-batches and shuffles on exit.

#### Tagged transforms and reweighting

`PandasDirDataModule` exposes `.transform`, `.tmp_transform`, `.reweight`,
`.copy`, `.merge`, `.rebatch_files`, and `.shuffle` for non-destructive
manipulation. Each appends a `tag` to `ds_info['tags']` so the modification
history travels with the dataset; `run_reweight_loop` relies on this to chain
`p_over_q_{i}` reweight columns. See `examples/example_reweight_loop.py` for
the end-to-end flow.
