# Driver anomaly detection from pose (DMD)

Code for the experiments on driver anomaly detection from YOLO26-pose skeletons on the
DMD dataset (leave-one-subject-out protocol, 14 drivers).

- `pipeline/` — clip extraction, pose detection, skeleton normalization, feature building
- `common/`, `training/`, `baselines/`, `aggregation/`, `eval/` — earlier experiments
- `paper_experiments/` — the LOSO runs reported in the article
  - `paper_loso.py`, `paper_extra.py`, `paper_mamba.py`, `paper_baselines.py` — autoencoders and flattened-window baselines
  - `paper_ablation.py` — feature / bottleneck / kernel-size ablations of the TCN autoencoder; `--pool mean` trains the autoencoder on the window-averaged vectors
  - `paper_classical.py` — one-class methods on window-level summaries of the features
  - `time_classical.py` — CPU time of OC-SVM and of the TCN autoencoder
- `analysis/` — scripts that turn the result files into the tables and figures (`bootstrap.py`: confidence intervals over clips, `pairs_tcn.py`: paired tests of the ablations, `fig_pipeline.py`: overview figure)
- `eval/driver_shift.py` — how the normal driving of each driver differs from the other drivers, per group of features

The DMD videos and everything derived from them (clips, skeletons, features, scores) are
not part of this repository; the dataset is available from https://dmd.vicomtech.org/
under its own license. The paths in the scripts point to the directory layout used on
the training server.
