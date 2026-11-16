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

The input is a pickled Python list of 1-D integer arrays, one per MD trajectory, where each entry is the state label (0 … n_states-1) of that frame


