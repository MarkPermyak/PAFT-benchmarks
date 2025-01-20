# PAFT-benchmarks

## Installing libraries:
```
pip install -r requiremets.txt
```

## (For PAFT only) Create a dictionary of optimal features permutations:

```
python hyfd/hyfd.py ./data/train
python hyfd/paft_fd_distilation_and_optimization.py
```

## Fit models without GPU support (ganblr, ganblr++, bayesian_network, ddpm) and generate synthetic data:
```
python benchmark_cpu/fit_and_generate.py
```

## Fit models with GPU support (great, paft, ctgan, tvae, rtvae, adsgan, pategan) and generate synthetic data:
```
python benchmark_gpu/fit_and_generate.py
```

## Evaluate benchmarks:
```
python benchmarks_cpu/synthcity_metrics.py
```