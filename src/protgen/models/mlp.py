from protgen.data import ProteinDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np 

train_data = ProteinDataset(path_to_data="/home/klaus/deep/project/02456-autoregressive-project/data/train.csv")
val_data = ProteinDataset(path_to_data="/home/klaus/deep/project/02456-autoregressive-project/data/val.csv")

n = int
k = int 
x = np.array(n, k)
x, x_shifted, y, length = train_data[0]
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


