# online-robustcpt
Implementation of the robust change point detection and robust mean testing algorithms from the paper **Online change point detection under heavy-tailedness and contamination**, together with Julia and R scripts for reproducing the simulations and plots.

## Main methods implemented

This repository implements:

- robust univariate and multivariate online change point detection algorithms;
- robust multivariate mean testing algorithms;
- Julia simulation scripts for studying these algorithms under heavy tails and contamination;
- R scripts for reproducing the numerical plots in the paper.

## Repository structure
```text
.
├── julia_utils/
│   └── hd_utils.jl
├── julia_multivariate_cpt/
│   ├── mvcpt_laplace.jl
│   └── mvcpt_tdist.jl
├── julia_multivariate_testing/
│   ├── mvtest_laplace_misspec.jl
│   ├── mvtest_laplace_varykn.jl
│   ├── mvtest_laplace_varykp.jl
│   ├── mvtest_tdist_misspec.jl
│   ├── mvtest_tdist_varykn.jl
│   └── mvtest_tdist_varykp.jl
├── julia_univariate_cpt/
│   ├── change_point_utils.jl
│   ├── univ_cpt_laplace.jl
│   └── univ_cpt_tdist.jl
├── R_plots/
│   ├── multiv_cpt_plot.R
│   ├── multiv_test_plot.R
│   ├── univ_laplce_plot.R
│   └── univ_tdist_plot.R
└── README.md
```

The folders are organised as follows:

- `julia_utils/` contains shared utilities for the multivariate algorithms.
- `julia_multivariate_cpt/` contains multivariate change point detection experiments.
- `julia_multivariate_testing/` contains multivariate robust mean testing experiments.
- `julia_univariate_cpt/` contains univariate change point detection algorithms and experiments.
- `R_plots/` contains R scripts for reproducing the plots in the paper.

## File naming convention

For the multivariate mean testing experiments:

- `misspec` refers to experiments studying type I error and power across different values of `kappa0` given a true value of `kappa`;
- `varykn` refers to experiments varying `n` with fixed `p`;
- `varykp` refers to experiments varying `p` with fixed `n`;
- `laplace` and `tdist` indicate the inlier distribution used in the experiment.


## Requirements

### Julia

Tested with Julia version `1.12.0`.

Required Julia packages include:

```julia
CSV
DataFrames
Distributed
LinearAlgebra
Random
Statistics
Distributions
```

### R

Tested with R version `4.5.2`.

Required R packages include:

```r
MASS
ggplot2
dplyr
tidyr
tibble
```

## Reproducing the figures
To reproduce a figure, first run the corresponding Julia script to generate the simulation output as a CSV file. Then specify the path to this CSV file in the relevant R plotting script.