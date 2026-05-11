# `iwpc.datasets`

`torch.utils.data.Dataset` implementations that turn pandas DataFrames (in memory or on disk as a list of pickles) into batches matching `iwpc`'s `(features, labels, weights)` training contract. These sit underneath the `data_modules/` Lightning DataModules and never normally need to be constructed by user code.

## Layout

- `pandas_dataset.py` — `PandasDataset` (wraps a single in-memory DataFrame), plus the `StructuredData` / `StructuredDataSpec` recursive type aliases and helpers `structure_data`, `recursive_slice_structured_data`, `recursive_share_memory`.
- `pandas_file_list_dataset.py` — `PandasFileListDataset` (lazily loads one pickled DataFrame at a time from a list of files; backs `PandasDirDataModule`).

## Feature spec

Both datasets take a `feature_spec: StructuredDataSpec`, a recursive list of column names. A flat spec emits one stacked tensor per sample; a nested spec emits a tuple of tensors. Example: `['__label', 'x', 'y']` would give one shape-`(3,)` tensor; `[['x', 'y'], '__label']` would give a tuple of shapes `(2,)` and `(1,)`. The `(features, label, weight)` triple consumed by estimators is produced by combining the spec with `weight_col`.

## Usage

In-memory single-DataFrame use:

```python
from iwpc.datasets.pandas_dataset import PandasDataset

ds = PandasDataset(df, feature_spec=['x', 'y', '__label'], weight_col='w')
features_and_weight = ds[0]                         # tuple of tensors
ds.share_memory_()                                  # for multi-worker loaders
```

On-disk, multi-file (typically only seen via `PandasDirDataModule`):

```python
from iwpc.datasets.pandas_file_list_dataset import PandasFileListDataset

ds = PandasFileListDataset(
    files=['file_0.pkl', 'file_1.pkl', 'file_2.pkl'],
    feature_spec=['x', 'y', '__label'],
    weight_col='w',
    file_sizes=[10000, 10000, 10000],   # optional; otherwise inferred by opening each file
    shuffle_in_file=True,
)
```

When using `PandasFileListDataset`, do **not** wrap it in a `DataLoader` with `shuffle=True`: random global indices would trigger one disk read per sample. Shuffling is performed inside each file via `shuffle_in_file`.
