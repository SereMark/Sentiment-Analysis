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
import ipywidgets as widgets
from IPython.display import display, clear_output
import threading
import time
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
    imdb_data['review'].tolist(), imdb_data['sentiment'].tolist(), test_size=0.2, random_state=42
)

# Separate reviews based on sentiment
positive_reviews = [len(review.split()) for review, sentiment in zip(imdb_data['review'].tolist(), imdb_data['sentiment'].tolist()) if sentiment == 1]
negative_reviews = [len(review.split()) for review, sentiment in zip(imdb_data['review'].tolist(), imdb_data['sentiment'].tolist()) if sentiment == 0]

# Plot histogram of review lengths for each sentiment
plt.figure()
plt.hist(positive_reviews, bins=30, color='blue', alpha=0.7, label='Positive')
plt.hist(negative_reviews, bins=30, color='red', alpha=0.7, label='Negative')
plt.title('Histogram of Review Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

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
bert_losses = train(bert_classifier, train_loader, optimizer_bert, loss_function, device, 'BERT', epochs=10)
glove_losses = train(glove_classifier, glove_train_loader, optimizer_glove, loss_function, device, 'GloVe', epochs=10)

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

# Training loss comparison function
def plot_combined_training_loss(losses1, losses2, name1, name2):
    plt.figure()
    plt.plot(losses1, label=f'Training Loss for {name1}')
    plt.plot(losses2, label=f'Training Loss for {name2}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

# Plot training loss comparison
plot_combined_training_loss(bert_losses, glove_losses, 'BERT', 'GloVe')

# Histogram of predicted probabilities comparison function
def plot_combined_probability_histogram(probs1, probs2, name1, name2):
    plt.figure()
    plt.hist(probs1, bins=10, alpha=0.75, label=f'Predicted Probabilities {name1}')
    plt.hist(probs2, bins=10, alpha=0.75, label=f'Predicted Probabilities {name2}')
    plt.title('Histogram of Predicted Probabilities')
    plt.xlabel('Probability of Positive Sentiment')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Plot histogram of predicted probabilities
plot_combined_probability_histogram(bert_probs, glove_probs, 'BERT', 'GloVe')

# ROC curve comparison function
def plot_combined_roc_curve(true_labels1, probs1, true_labels2, probs2, name1, name2):
    fpr1, tpr1, _ = roc_curve(true_labels1, probs1)
    roc_auc1 = auc(fpr1, tpr1)
    fpr2, tpr2, _ = roc_curve(true_labels2, probs2)
    roc_auc2 = auc(fpr2, tpr2)
    
    plt.figure()
    plt.plot(fpr1, tpr1, label=f'{name1} ROC curve (area = {roc_auc1:.2f})')
    plt.plot(fpr2, tpr2, label=f'{name2} ROC curve (area = {roc_auc2:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curve comparison
plot_combined_roc_curve(bert_labels, bert_probs, glove_labels, glove_probs, 'BERT', 'GloVe')

# Confusion matrix comparison function
def plot_combined_confusion_matrix(true_labels1, predictions1, true_labels2, predictions2, name1, name2):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    cf_matrix1 = confusion_matrix(true_labels1, predictions1)
    sns.heatmap(cf_matrix1, annot=True, fmt='g', ax=ax[0])
    ax[0].set_title(f'Confusion Matrix for {name1}')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    
    cf_matrix2 = confusion_matrix(true_labels2, predictions2)
    sns.heatmap(cf_matrix2, annot=True, fmt='g', ax=ax[1])
    ax[1].set_title(f'Confusion Matrix for {name2}')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Actual')
    
    plt.show()

# Plot confusion matrix comparison
plot_combined_confusion_matrix(bert_labels, bert_preds, glove_labels, glove_preds, 'BERT', 'GloVe')

# Precision-Recall curve comparison function
def plot_combined_precision_recall_curve(true_labels1, probs1, true_labels2, probs2, name1, name2):
    precision1, recall1, _ = precision_recall_curve(true_labels1, probs1)
    precision2, recall2, _ = precision_recall_curve(true_labels2, probs2)
    
    plt.figure()
    plt.plot(recall1, precision1, label=f'{name1} Precision-Recall curve')
    plt.plot(recall2, precision2, label=f'{name2} Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend()
    plt.show()

# Plot precision-recall curve comparison
plot_combined_precision_recall_curve(bert_labels, bert_probs, glove_labels, glove_probs, 'BERT', 'GloVe')

# Sentiment prediction function
def predict_sentiment(text, model, tokenizer, device):
    if not text.strip():  # Check if the text is not just empty or spaces
        return "No input", 0
    model.eval()  # Ensure the model is in evaluation mode
    text = text.lower()  # Lowercase the text
    try:
        if tokenizer:
            tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(tokenized['input_ids'], tokenized['attention_mask']).softmax(dim=-1)
        else:
            embeddings = fetch_glove_embeddings(text).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(embeddings).softmax(dim=-1)
        pred = torch.argmax(output, dim=1).item()
        certainty = output[0, pred].item()
        sentiment = 'Positive' if pred == 1 else 'Negative'
        return sentiment, certainty * 100
    except Exception as e:
        return f"Error: {str(e)}", 0

# Sentiment prediction widget for google colab
def update_output(change):
    text_input = change.new
    if text_input:  # Update only if the text is not empty
        bert_sentiment, bert_certainty = predict_sentiment(text_input, bert_classifier, bert_tokenizer, device)
        glove_sentiment, glove_certainty = predict_sentiment(text_input, glove_classifier, None, device)
        with output:
            clear_output(wait=True)
            print(f"BERT: Sentiment - {bert_sentiment}, Certainty - {bert_certainty:.2f}%")
            print(f"GloVe: Sentiment - {glove_sentiment}, Certainty - {glove_certainty:.2f}%")

# Widget setup
text_input = widgets.Text(placeholder='Type something here...', description='Input:', disabled=False)
output = widgets.Output()
text_input.observe(update_output, names='value')
display(text_input, output)