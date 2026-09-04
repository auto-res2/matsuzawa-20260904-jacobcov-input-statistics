# Committed inputs

Both files are small, fixed extracts of public benchmarks so that every run is
self-contained (no cluster cache, no download) and the commit hash alone fixes
the inputs.

- `nb201_cifar10_subset.json` — NAS-Bench-201 (Dong & Yang, ICLR 2020) CIFAR-10
  test accuracy (`eval_acc1es`, %) for the 1,000 cells sampled uniformly without
  replacement from the sorted list of all 15,625 cell strings with
  `numpy.random.default_rng(2026)`. Identical to the architecture set of the
  2026-09-02/03 studies.
- `cifar10_batches.npz` — for each seed s in {0,1,2,10,11,12}, the 128 CIFAR-10
  training images (`data_batch_1`, uint8, (128,3,32,32)) selected by
  `numpy.random.default_rng(s).choice(10000, 128, replace=False)`, plus the
  indices. Labels are not stored: no proxy in this study uses labels.
