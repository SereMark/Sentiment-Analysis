import pandas as pd
import torch

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
dataset = pd.read_csv('IMDB Dataset.csv')