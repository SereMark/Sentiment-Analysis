import pandas as pd
import torch
import torch.nn as nn
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torchtext.vocab as vocab
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import warnings

# Disable 1Torch warning
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare dataset
dataset = pd.read_csv('IMDB Dataset.csv')
dataset['review'] = dataset['review'].str.lower()  # Convert to lowercase
dataset['sentiment'] = dataset['sentiment'].map({'positive': 1, 'negative': 0})  # Convert labels to binary

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    dataset['review'].tolist(),
    dataset['sentiment'].tolist(),
    test_size=0.2,
    random_state=42
)

# GloVe embeddings setup
glove = vocab.GloVe(name='6B', dim=100)  # 100-dimensional GloVe vectors

def get_glove_embedding(text):
    tokens = text.split()
    embeddings = [glove[t].unsqueeze(0) for t in tokens if t in glove.stoi]
    return torch.cat(embeddings, dim=0).mean(dim=0) if embeddings else torch.zeros(glove.dim)

train_embeddings = [get_glove_embedding(text) for text in train_texts]
val_embeddings = [get_glove_embedding(text) for text in val_texts]

# Dataset class for GloVe embeddings
class GloVeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings, self.labels = embeddings, labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            'embedding': torch.tensor(self.embeddings[idx], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# BERT tokenizer and model setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MovieReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized_data = tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized_data['input_ids'].squeeze(),
            'attention_mask': tokenized_data['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Model for sentiment classification
class SentimentClassifier(nn.Module):
    def __init__(self, model_type='BERT'):
        super().__init__()
        if model_type == 'BERT':
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.dropout = nn.Dropout(0.3)
            self.out = nn.Linear(self.model.config.hidden_size, 2)
        else:  # GloVe
            self.fc1 = nn.Linear(100, 50)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(50, 2)

    def forward(self, x, attention_mask=None):
        if hasattr(self, 'model'):
            output = self.model(x, attention_mask=attention_mask, return_dict=False)[1]
            output = self.dropout(output)
            return self.out(output)
        else:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            return self.fc2(x)

# Training and evaluation functions
def train(model, loader, optimizer, criterion, device, model_type='BERT'):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        optimizer.zero_grad()
        if model_type == 'BERT':
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
        else:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Training Loss: {total_loss / len(loader)}")

def evaluate(model, loader, device, model_type='BERT'):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if model_type == 'BERT':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].numpy()
                outputs = model(input_ids, attention_mask)
            else:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].numpy()
                outputs = model(embeddings)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels)
    print(classification_report(true_labels, predictions))

# Instantiate and setup DataLoaders for both models
train_loader = DataLoader(MovieReviewDataset(train_texts, train_labels), batch_size=16, shuffle=True)
val_loader = DataLoader(MovieReviewDataset(val_texts, val_labels), batch_size=16, shuffle=False)
glove_train_loader = DataLoader(GloVeDataset(train_embeddings, train_labels), batch_size=16, shuffle=True)
glove_val_loader = DataLoader(GloVeDataset(val_embeddings, val_labels), batch_size=16, shuffle=False)

# Instantiate models
bert_model = SentimentClassifier('BERT').to(device)
glove_model = SentimentClassifier('GloVe').to(device)
optimizer_bert = torch.optim.Adam(bert_model.parameters(), lr=2e-5)
optimizer_glove = torch.optim.Adam(glove_model.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss()

# Train and evaluate both models
train(bert_model, train_loader, optimizer_bert, criterion, device, 'BERT')
evaluate(bert_model, val_loader, device, 'BERT')
train(glove_model, glove_train_loader, optimizer_glove, criterion, device, 'GloVe')
evaluate(glove_model, glove_val_loader, device, 'GloVe')