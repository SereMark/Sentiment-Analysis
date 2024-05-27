import pandas as pd
import torch
import torch.nn as nn
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torchtext.vocab as vocab
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Suppress specific warnings from torch
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")

# Set the computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the dataset
imdb_data = pd.read_csv('IMDB Dataset.csv')
imdb_data['review'] = imdb_data['review'].str.lower()
imdb_data['sentiment'] = imdb_data['sentiment'].map({'positive': 1, 'negative': 0})

# Split the dataset into training and validation sets
train_reviews, val_reviews, train_sentiments, val_sentiments = train_test_split(
    imdb_data['review'], imdb_data['sentiment'], test_size=0.2, random_state=42
)

# Initialize GloVe embeddings
glove_embeddings = vocab.GloVe(name='6B', dim=100)

# Function to fetch GloVe embeddings for a text
def fetch_glove_embeddings(text):
    tokens = text.split()
    embeddings = [glove_embeddings[t].unsqueeze(0) for t in tokens if t in glove_embeddings.stoi]
    return torch.cat(embeddings, dim=0).mean(dim=0) if embeddings else torch.zeros(glove_embeddings.dim)

train_glove_embeddings = [fetch_glove_embeddings(text) for text in train_reviews]
val_glove_embeddings = [fetch_glove_embeddings(text) for text in val_reviews]

# Dataset class for GloVe embeddings
class GloVeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.stack(embeddings)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {'embedding': self.embeddings[idx].clone().detach(), 'label': self.labels[idx]}

# BERT tokenizer and model setup
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset class for BERT tokenized data
class BertDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized_data = bert_tokenizer(
            self.texts[idx], padding='max_length', truncation=True, max_length=256, return_tensors="pt"
        )
        return {
            'input_ids': tokenized_data['input_ids'].squeeze().clone().detach(),
            'attention_mask': tokenized_data['attention_mask'].squeeze().clone().detach(),
            'labels': self.labels[idx]
        }

# Generalized model class for sentiment analysis
class SentimentClassifier(nn.Module):
    def __init__(self, model_type='BERT'):
        super().__init__()
        if model_type == 'BERT':
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(self.model.config.hidden_size, 2)
        else:
            self.fc1 = nn.Linear(100, 50)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(50, 2)

    def forward(self, x, attention_mask=None):
        if hasattr(self, 'model'):
            output = self.model(x, attention_mask=attention_mask)[1]
            output = self.dropout(output)
            return self.classifier(output)
        else:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            return self.fc2(x)

# Load data into DataLoader
train_loader = DataLoader(BertDataset(train_reviews, train_sentiments), batch_size=16, shuffle=True)
val_loader = DataLoader(BertDataset(val_reviews, val_sentiments), batch_size=16, shuffle=False)
glove_train_loader = DataLoader(GloVeDataset(train_glove_embeddings, train_sentiments), batch_size=16, shuffle=True)
glove_val_loader = DataLoader(GloVeDataset(val_glove_embeddings, val_sentiments), batch_size=16, shuffle=False)

# Initialize models, optimizers, and loss function
bert_classifier = SentimentClassifier('BERT').to(device)
glove_classifier = SentimentClassifier('GloVe').to(device)
optimizer_bert = torch.optim.Adam(bert_classifier.parameters(), lr=2e-5)
optimizer_glove = torch.optim.Adam(glove_classifier.parameters(), lr=5e-4)
loss_function = nn.CrossEntropyLoss()

# Training function
def train(model, loader, optimizer, loss_function, device, model_type='BERT', epochs=1):
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
            else:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].to(device)
                outputs = model(embeddings)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        all_epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed. Training Loss: {avg_loss}")
    print("Training completed.")
    return all_epoch_losses

# Train both models
bert_losses = train(bert_classifier, train_loader, optimizer_bert, loss_function, device, 'BERT', epochs=1)
glove_losses = train(glove_classifier, glove_train_loader, optimizer_glove, loss_function, device, 'GloVe', epochs=1)

# Evaluation function
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
            else:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].to(device)
                outputs = model(embeddings)
                outputs = outputs.softmax(dim=-1)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(outputs.cpu().numpy()[:, 1])
    print("Evaluation completed.")
    print(classification_report(true_labels, predictions))
    return predictions, true_labels, probabilities

# Evaluate both models
bert_preds, bert_labels, bert_probs = evaluate(bert_classifier, val_loader, device, 'BERT')
glove_preds, glove_labels, glove_probs = evaluate(glove_classifier, glove_val_loader, device, 'GloVe')

# Method to plot training losses
def plot_training_loss(losses, model_name):
    plt.figure()
    plt.plot(losses, label=f'Training Loss for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Epochs for {model_name}')
    plt.legend()
    plt.show()

# Plot Training Losses
plot_training_loss(bert_losses, 'BERT')
plot_training_loss(glove_losses, 'GloVe')

# Method to plot histograms of predicted probabilities
def plot_probability_histogram(probs, model_name):
    plt.figure()
    plt.hist(probs, bins=10, alpha=0.75, label=f'Predicted Probabilities {model_name}')
    plt.title(f'Histogram of Predicted Probabilities for {model_name}')
    plt.xlabel('Probability of Positive Sentiment')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Plot Histograms of Predicted Probabilities
plot_probability_histogram(bert_probs, 'BERT')
plot_probability_histogram(glove_probs, 'GloVe')

# Method to plot ROC curves
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

# Plot ROC Curves
plot_roc_curve(bert_labels, bert_probs, 'BERT')
plot_roc_curve(glove_labels, glove_probs, 'GloVe')

# Method to plot confusion matrices
def plot_confusion_matrix(true_labels, predictions, model_name):
    cf_matrix = confusion_matrix(true_labels, predictions)
    sns.heatmap(cf_matrix, annot=True, fmt='g')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot Confusion Matrices
plot_confusion_matrix(bert_labels, bert_preds, 'BERT')
plot_confusion_matrix(glove_labels, glove_preds, 'GloVe')

# Method to plot Precision-Recall curves
def plot_precision_recall_curve(true_labels, model_probs, model_name):
    precision, recall, _ = precision_recall_curve(true_labels, model_probs)
    plt.figure()
    plt.plot(recall, precision, marker='.', label=f'Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend()
    plt.show()

# Plot Precision-Recall Curves
plot_precision_recall_curve(bert_labels, bert_probs, 'BERT')
plot_precision_recall_curve(glove_labels, glove_probs, 'GloVe')