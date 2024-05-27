import pandas as pd
import torch
import torch.nn as nn
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torchtext.vocab as vocab
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
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
        self.embeddings = torch.stack(embeddings)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx].clone().detach(),
            'label': self.labels[idx]
        }

# BERT tokenizer and model setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MovieReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)

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
            'input_ids': tokenized_data['input_ids'].squeeze().clone().detach(),
            'attention_mask': tokenized_data['attention_mask'].squeeze().clone().detach(),
            'labels': self.labels[idx]
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
def train(model, loader, optimizer, criterion, device, model_type='BERT', epochs=1):
    model.train()
    all_epoch_losses = []
    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        print(f"Epoch {epoch+1}/{epochs} started...")
        for batch in tqdm(loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            if model_type == 'BERT':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                print("Processed BERT model inputs.")
            else:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].to(device)
                outputs = model(embeddings)
                print("Processed non-BERT model inputs.")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        all_epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed. Training Loss: {avg_loss}")
    print("Training completed.")
    return all_epoch_losses

def evaluate(model, loader, device, model_type='BERT'):
    model.eval()
    predictions, true_labels, probabilities = [], [], []
    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if model_type == 'BERT':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                outputs = outputs.softmax(dim=-1)
                print("Processed BERT model evaluation.")
            else:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].to(device)
                outputs = model(embeddings)
                outputs = outputs.softmax(dim=-1)
                print("Processed non-BERT model evaluation.")
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(outputs.cpu().numpy()[:, 1])
    print("Evaluation completed.")
    print(classification_report(true_labels, predictions))
    return predictions, true_labels, probabilities

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

# Train both models
bert_losses = train(bert_model, train_loader, optimizer_bert, criterion, device, 'BERT', epochs=3)
glove_losses = train(glove_model, glove_train_loader, optimizer_glove, criterion, device, 'GloVe', epochs=10)

# Evaluate both models
bert_preds, bert_labels, bert_probs = evaluate(bert_model, val_loader, device, 'BERT')
glove_preds, glove_labels, glove_probs = evaluate(glove_model, glove_val_loader, device, 'GloVe')

def plot_training_loss(losses, model_name):
    plt.figure()
    plt.plot(losses, label=f'Training Loss for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Epochs for {model_name}')
    plt.legend()
    plt.show()

def plot_probability_histogram(probs, model_name):
    plt.figure()
    plt.hist(probs, bins=10, alpha=0.75, label=f'Predicted Probabilities {model_name}')
    plt.title(f'Histogram of Predicted Probabilities for {model_name}')
    plt.xlabel('Probability of Positive Sentiment')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_roc_curve(true_labels, model_probs, model_name):
    fpr, tpr, _ = roc_curve(true_labels, model_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(true_labels, predictions, model_name):
    cf_matrix = confusion_matrix(true_labels, predictions)
    sns.heatmap(cf_matrix, annot=True, fmt='g')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_precision_recall_curve(true_labels, model_probs, model_name):
    precision, recall, _ = precision_recall_curve(true_labels, model_probs)
    plt.figure()
    plt.plot(recall, precision, marker='.', label=f'Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend()
    plt.show()

# Assuming you capture losses per epoch in a list:
plot_training_loss(bert_losses, 'BERT')
plot_training_loss(glove_losses, 'GloVe')

# Plot Histograms of Predicted Probabilities
plot_probability_histogram(bert_probs, 'BERT')
plot_probability_histogram(glove_probs, 'GloVe')

# Plot ROC Curves
plot_roc_curve(bert_labels, bert_probs, 'BERT')
plot_roc_curve(glove_labels, glove_probs, 'GloVe')

# Plot Confusion Matrices
plot_confusion_matrix(bert_labels, bert_preds, 'BERT')
plot_confusion_matrix(glove_labels, glove_preds, 'GloVe')

# Plot Precision-Recall Curves
plot_precision_recall_curve(bert_labels, bert_probs, 'BERT')
plot_precision_recall_curve(glove_labels, glove_probs, 'GloVe')