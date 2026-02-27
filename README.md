# Learning Quantum Data Distribution via Chaotic Quantum Diffusion ModelUniversality of Many-body Projected Ensemble for Learning Quantum Data Distribution

## Introduction

This repository contains the code implementation for the paper *"Learning Quantum Data Distribution via Chaotic Quantum Diffusion Model"* [arXiv:2602.22061](https://arxiv.org/abs/2602.22061). The codebase provides tools and scripts to replicate the experiments described in the paper, focusing on learning quantum data distributions using the chaotic quantum diffusion framework.

## Repository Structure

The codebase is organized into the following modules:

- **data**: Utility functions for generating quantum data used in the experiments.
- **model**: JAX-based implementation of the chaotic quantum diffusion (QDM) class, including training utility functions.
- **utils**: Utility functions for implementing quantum circuits using TensorCircuit, computing Wasserstein and MMD distances, and calculating the Vendi score.
- **main**: Scripts `main_gen_demo.py` and `main_gen_mol_uncon.py` for generating multi-clustered quantum states and QM9 quantum states, respectively.
- **datasets**: Directory containing filtered data from the QM9 dataset.
- **postprocess**: Functions for plotting and analyzing experimental results.
- **runscripts**: Shell scripts to execute the experiments.

## Prerequisites

To run the code, ensure the following requirements are met:

- Python version <= 3.10.12
- JAX version <= 0.6.1
- Additional dependencies are listed in `requirements.txt`.

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Running the Experiments

Two scripts are provided to replicate the experiments described in the paper:

1. **Multi-cluster Quantum States Ensemble**:
   To train the model on multi-cluster quantum states, execute the following scripts.

Using RUCD:

```bash
sh runscripts_demo/demo_train_multi_cluster_RUCD.sh
```

Using CTED/RTED:

```bash
sh runscripts_demo/demo_train_multi_cluster_chaotic.sh
```

2. **Circular Quantum States Ensemble**:
   To train the model on circular quantum states, execute the following scripts.

Using RUCD:

```bash
sh runscripts_demo/demo_train_circular_RUCD.sh
```

Using CTED/RTED:

```bash
sh runscripts_demo/demo_train_circular_chaotic.sh
```

3. **QM9 Dataset**:
   To train the model on the QM9 dataset, execute the following scripts.

Using RUCD:

```bash
sh runscripts_demo/demo_train_QM9_mol_8_2_full_RUCD.sh
```

Using CTED/RTED:

```bash
sh runscripts_demo/demo_train_QM9_8_2_full_chaotic.sh
```

If you want to train RUCD, CTED/RTED in the latent space (reduced by Quantum Auto Encoder) run the following scripts:

```bash
sh runscripts_demo/demo_train_QM9_8_2_reduced_chaotic.sh
sh runscripts_demo/demo_train_QM9_8_2_reduced_RUCD.sh
```

4. **Investigate the noise robustness**:

Noise model in RUCD:

```bash
sh runscripts_demo/demo_train_RUCD_noise.sh
```

Noise model in CTED/RTED:

```bash
sh runscripts_demo/demo_train_chaotic_noise.sh
```

## Post-processing Results

To visualize and analyze the experimental results, navigate to the `postprocess` directory and run the following scripts:

- For multi-cluster quantum states (change the folder to plot the circular ensemble):
  
  ```bash
  python postprocess/plot_compare_diff_EVAL_multi_cluster.py
  ```


- For QM9 dataset:

```bash
python postprocess/plot_compare_QM9_8_2_full_latent_1.py
```

- For noise model:

```bash
python postprocess/plot_compare_noise_multi_cluster_1.p
```

## Contributing

We welcome contributions to enhance the codebase. Please submit pull requests or open issues to suggest improvements, report bugs, or add new features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

