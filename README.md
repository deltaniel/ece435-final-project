# Safe RLHF Replication

> **Course:** COS 435

> **Authors:** Daniel Ruan, Peter Kirgis, and Myles Anderson

> **Date:** May 11, 2025

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Requirements](#requirements)  
4. [References](#references)  

---

## Project Overview

This is an attempt to replicate the methodology and experiments from “Safe Reinforcement Learning from Human Feedback” (Dai et al., 2024).  Our replication utilizes the reward and cost models trained by the original authors, but re-writes the central Safe RLHF method from scratch. 

The key result from the original paper that we try to reproduce are win rates calculated by GPT-4 on helpfulness and harmfulness between Alpaca-7B, a traditional RLHF finetuned model, and a Safe RLHF model.

## Repository Structure

The repository is organized as follows:

```
├── eval
│   ├── eval_results.ipynb  # Jupyter notebook for evaluating the model
│   └── generate_answers.py # Script for generating answers
│  
├── job_scripts # Bash scripts for running jobs on Princeton's Della cluster
│  
├── scripts # Helper scripts for memory and usage
|
└── src
    ├── dataloader.py # Data loading and preprocessing
    ├── ppo_vanilla.py # Traditional RLHF implementation
    └──  safe_rlhf.py # Safe RLHF implementation
```

## Requirements

To run the code, install all packages by running 
```bash
pip install -e .
```
from the root repository directory.

## References
- Dai, J., Pan, X., Sun, R., Ji, J., Xu, X., Liu, M., ... & Yang, Y. (2023). Safe rlhf: Safe reinforcement learning from human feedback. arXiv preprint arXiv:2310.12773.

