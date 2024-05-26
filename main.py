import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import warnings

# Ignore unimportant warnings
warnings.filterwarnings("ignore", message="Torch was not compiled with CUDA enabled")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
dataset = pd.read_csv('IMDB Dataset.csv')

# Simple preprocessing
dataset['review'] = dataset['review'].str.lower()  # Convert to lowercase
dataset['sentiment'] = dataset['sentiment'].map({'positive': 1, 'negative': 0})  # Convert labels to binary

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    dataset['review'].tolist(),
    dataset['sentiment'].tolist(),
    test_size=0.2,
    random_state=42
)

class MovieReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized_data = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized_data['input_ids'].squeeze()
        attention_mask = tokenized_data['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create DataLoaders
train_dataset = MovieReviewDataset(train_texts, train_labels, tokenizer)
val_dataset = MovieReviewDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        return self.out(dropout_output)

# Model instantiation
model = SentimentClassifier().to(device)

# Training function
def train(model, train_loader, val_loader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1):
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# Call the training function
train(model, train_loader, val_loader, device)

def evaluate(model, val_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels)

    print(classification_report(true_labels, predictions))

# Evaluate the model
evaluate(model, val_loader, device)