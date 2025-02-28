# Multi-Agent Safe Bayesian Optimization on the Quasar Platform

## Overview

This repository contains the implementation of my Master's thesis "Learning-based control of multi-agent systems", which extends the SafeOpt algorithm to a multi-agent setting, utilizing the Quasar platform for experimental validation. The work focuses on safe and efficient optimization in a decentralized cooperative control scenario.

## Key Contributions

- **Extension of SafeOpt**: Expands the SafeOpt algorithm to a multi-agent environment.
- **Multi-Agent Safe Learning**: Implements decentralized safe learning for multiple agents interacting in a shared control task.
- **Quanser Platform Integration**: Deploys and evaluates the approach on the Quanser platform for real-world experimental validation.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/DimitrisGkoutzounis/QUANCER_MASAFEOPT.git
cd QUANCER_MASAFEOPT
pip install matplotlib scipy numpy safeopt GPy
```

## Usage

Run the main optimization script:

```bash
python respective-experiment.py
```


## Citation

If you use this work, please cite:

```
@mastersthesis{gkoutzounis2025,
  author = {Dimitrios Gkoutzounis},
  title = {Learning-based control of multi-agent systems},
  school = {Aalto University & Hellenic Mediterranean University},
  year = {2025}
}
```

