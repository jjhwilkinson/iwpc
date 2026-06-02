# AGENTS.md — `iwpc.data_modules`

## Batch contract (non-negotiable)

- Every dataloader here yields `(features, labels, weights)` tuples.
- `labels == 0` -> samples from `p`. `labels == 1` -> samples from `q`.
- The pandas/numpy binary modules add this label automatically.
  `BinaryPandasDataModule` writes a 0/1 column whose name is taken from
  `feature_spec[1][0]` (the first target column in the nested
  `[feature_cols, [label_col, ...]]` convention shared with
  `PandasDirDataModule`); `BinaryNumpyDataModule` concatenates a 0/1 tensor.
  For `PandasDirDataModule` the label column is whatever you point
  `feature_spec` at as a target — sample files in `examples/sample_dataset/`
  use `'label'`.
- Weights in `BinaryNumpyDataModule` are mean-normalised within each class.
  `PandasDirDataModule.normalise_weights(label_col=...)` does the same for
  on-disk datasets and is automatically called by `.reweight`.

## Train/val split mechanics

- `PandasDataModule` / `BinaryPandasDataModule` / `BinaryNumpyDataModule`:
  `sklearn.train_test_split(shuffle=True)` over rows, default 50/50.
- `PandasDirDataModule`: split is **by file**, not by row. First
  `ceil(num_files * split)` files become `train_files`, the rest become
  `validation_files`. On-disk ordering matters — call `.shuffle()` or build
  with `PandasDirDataModuleBuilder(shuffle=True)` to randomise across files.
- `use_in_memory_dataset=True` concatenates all train/val files into a single
  `PandasDataset` with shared-memory tensors during `setup()`; the train
  loader then gets `shuffle=True` and `pin_memory=True` defaults (both
  overridable via `dataloader_kwargs`). Only viable when the dataset fits in
  RAM × num_workers.

## `ds_info.yml` fields

- `file_sizes` (required): ordered list of row counts per `file_i.pkl`.
- `tags` (optional): append-only history of modifications.
  `PandasDirDataModuleBuilder` prepends a `"Created: <iso timestamp>"` tag.
- Anything else you put in is preserved across `.transform` unless explicitly
  overridden via `new_ds_info=` / `update_ds_info=`. `tags` cannot be
  overwritten, only appended.

## Mutation API (PandasDirDataModule only)

- `.transform(fn, out_dir, tag=..., force=...)`: per-file map, writes to a
  temp directory first then atomically moves into `out_dir`, recomputes
  `file_sizes`, appends `tag`. Returns a fresh `PandasDirDataModule`.
- `.tmp_transform(fn)`: same but yields a data module backed by a temp dir
  that is removed on context exit.
- `.reweight(tag, reweight_fn, out_dir, ...)`: multiplies the weight column by
  `reweight_fn(df)`, then `normalise_weights`. Requires either `self.weight_col`
  or an explicit `output_weight_col`.
- `.copy(**overrides)`: rebuilds the data module with the same kwargs.
  `run_reweight_loop` uses this to swap `weight_col` after a reweight.
- `.merge`, `.rebatch_files`, `.shuffle`: bulk reshape; all append tags.

## Cross-package consumers

- `iwpc.calculate_divergence.calculate_divergence` accepts any of these data
  modules — it only needs `train_dataloader` / `val_dataloader`.
- `iwpc.reweight_loop.run_reweight_loop` **only** works with
  `PandasDirDataModule`: it calls `.copy`, `.transform`, and writes
  `p_over_q_{i}` columns into the on-disk dataset between iterations.
- The actual `Dataset` plumbing lives in `iwpc.datasets`
  (`PandasDataset`, `PandasFileListDataset`, `StructuredDataSpec`); don't
  bypass it.
- `examples/example_reweight_loop.py` + `examples/sample_dataset/` are the
  canonical reference for the on-disk layout.
