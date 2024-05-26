import pandas as pd
import torch

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
df = pd.read_csv('C:/Users/serem/Documents/Workspaces/Sentiment Analysis/IMDB Dataset.csv')