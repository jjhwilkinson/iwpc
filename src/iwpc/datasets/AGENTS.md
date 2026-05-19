# AGENTS — `iwpc.datasets`

- `PandasDataset(df, feature_spec, weight_col=None)` — converts columns into tensors eagerly via `structure_data` and stores them in `self.structured_data`. Appends a weight tensor (`ones` if `weight_col is None`) onto the end so `__getitem__` always yields `(..., weight)`. `share_memory_()` recursively calls `share_memory_()` on every tensor — used by `PandasDirDataModule(use_in_memory_dataset=True)`.
- `PandasFileListDataset(files, feature_spec, weight_col=None, file_sizes=None, shuffle_in_file=False)` — single-file LRU cache (`_last_file_no` / `_current_ds`). `__getitem__` resolves a global idx to `(file_idx, in_file_idx)` linearly via `file_and_in_file_idx`. `__len__` is `sum(file_sizes)`.
- Type aliases: `StructuredData = list[Tensor | list[...]]`, `StructuredDataSpec = list[str | list[...]]`. Helpers `structure_data`, `recursive_slice_structured_data`, `recursive_share_memory` operate on these.

## Subclass contract

- No formal contract; both classes are concrete. Anything claiming to be a `PandasDataset`-compatible producer must return tuples that end with a weight tensor, since downstream estimators (e.g. `NaiveVariationalFDivergenceEstimator`) unpack the last element as weights.
- `feature_spec` leaf check: `structure_data` treats a list as "leaf" iff every entry is a `str`. Mixing strings with sublists inside one bracket level will recurse, not stack — be deliberate about nesting.

## Cross-package consumers

- `src/iwpc/data_modules/pandas_data_module.py` — `BinaryPandasDataModule` builds `PandasDataset` for train and val splits.
- `src/iwpc/data_modules/pandas_directory_data_module.py` — wraps `PandasFileListDataset` for train/val/all; optionally concatenates files into an in-memory `PandasDataset.share_memory_()`.
- `src/iwpc/reweight_loop.py` — instantiates `PandasDataset` directly when running the reweight inference pass.

## Gotchas

- `PandasDataset.num_features` references `self.feature_cols`, which **does not exist** (only `feature_spec` is stored). Calling that property will `AttributeError`. Nothing in the codebase calls it; treat as dead code.
- `PandasFileListDataset` is **not safe for `DataLoader(shuffle=True)`**: adjacent indices from different files will thrash IO. The DataModule layer enforces `shuffle=False`.
- `pd.read_pickle` runs once per file load; if the pickle was written by a different pandas version, this will be the point of failure.
- `shuffle_in_file=True` shuffles only within the current file; it does **not** persist across reloads (each `load_file` re-shuffles).
- `share_memory_()` must be called before forking DataLoader workers, otherwise per-worker copies will be made.
