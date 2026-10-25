# RENA v9 Milestone Report — Multi-Architecture CIFAR-10 Benchmark Path

## Implemented
- Added CIFAR-adapted ResNet-18 trained from scratch alongside the existing compact CNN.
- CIFAR runner now supports `cnn`, `resnet18`, or `both` in one reproducible run.
- Separated candidate model training from the reference ensemble using disjoint seed ranges. The previous implementation incorrectly used the first reference as the candidate, which would have contaminated candidate/reference comparisons.
- Added split SHA-256, split sizes, architecture labels, explicit device setting, deterministic DataLoader shuffle generator, and candidate/reference seed-disjointness metadata.
- Exposed architecture/device selection in the research API and command-line runner.
- Added CPU forward-pass tests for both architectures and validation tests.

## Validation performed
- `python3 -m compileall -q core backend tests` — passed.
- `PYTHONPATH=. python3 -m unittest tests.test_cifar_architectures -v` — 4/4 passed.
- `PYTHONPATH=. python3 -m unittest tests.test_benchmark_milestone tests.test_dependence tests.test_api_schema -q` — 9 tests reported passed in 32.648 seconds, but the process did not terminate cleanly before the outer timeout. Do not interpret this as a clean full-suite run.
- No CIFAR training result is claimed in this milestone: downloading CIFAR-10 was previously blocked by network/DNS. This milestone validates the benchmark implementation and model interfaces, not empirical CIFAR findings.

## Run on a machine with CIFAR-10 download access

```bash
python3 scripts_run_cifar_benchmark.py 2 3 both
```

This runs 2 epochs per model and architecture, with one candidate plus three independent reference retrains for each architecture. For a publication-quality study, increase epochs/reference count and add repeated data splits, multiple forget fractions, actual unlearning methods, and held-out attacks.

## Scientific caveat
A shared retain split and independent initialization seeds measure variability across model training, not variability across data samples. This is a benchmark scaffold, not evidence that RENA outperforms another verifier. No ULTRA reproduction or performance superiority is claimed.
