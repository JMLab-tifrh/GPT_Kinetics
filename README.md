# Transformer-augmented Markov state models: PCSK9–peptide recognition

A decoder-only (GPT-style) transformer is trained on discretised MD trajectories (sequences of Markov-state labels) of the PCSK9–Pep2-8 system. It is then used to generate new state trajectories autoregressively, seeded on the first block_size states of each real trajectory.
