import pickle
from pathlib import Path

import torch

p = Path("new_splits\\validation_split.pkl")
with p.open("rb") as f:
    relative_beatmap_files = pickle.load(f)
print(relative_beatmap_files[0])


val_data = torch.load(Path("splits_mini_data\\test_data.pt"))
print(len(val_data))
