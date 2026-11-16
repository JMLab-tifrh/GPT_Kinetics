# Transformer-augmented Markov state models: PCSK9–peptide recognition

A decoder-only (GPT-style) transformer is trained on discretised MD trajectories (sequences of Markov-state labels) of the PCSK9–Pep2-8 system. It is then used to generate new state trajectories autoregressively, seeded on the first block_size states of each real trajectory.

# Repository Layout

```
.
├── gpt2_model/
│   └── gpt2.py                 # GPT model: attention heads, blocks, generate()
├── TRAIN_MODEL.ipynb           # data split (70/15/15 by trajectory) + training
├── INFER_MODEL.ipynb           # generate trajectories from the trained model
├── requirements.txt
└── LICENSE
```

# Installation

```
git clone https://github.com/<your-username>/pcsk9-transformer-msm.git
cd pcsk9-transformer-msm
conda create -n tch python=3.10 -y
conda activate tch
pip install -r requirements.txt
```


# Data

The input is a pickled Python list of 1-D integer arrays, one per MD trajectory, where each entry is the state label (0 … n_states-1) of that frame. The notebooks contain the placeholder path /path/to/pcsk9_gmm_nstates_{n_states}.pkl; point it to wherever you saved the file.

# Usage

## 1. Train

### Run TRAIN_MODEL.ipynb. 

It drops trajectories shorter than block_size + 1 frames, splits the remaining trajectories 70/15/15 into train/val/test  and saves the splits to SPLITS/, trains the GPT and saves the checkpoint with the lowest validation loss.

## 2. Generate trajectories

### Run INFER_MODEL.ipynb. 
For each trajectory longer than block_size, the model is seeded with its first "block_size" (context length) states and generates length_factor × len(trajectory) new states.



