import pandas as pd
import numpy as np
import torch
from torch import nn, optim, utils
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, matthews_corrcoef, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
from gensim import downloader as api
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

# Ignore flash attention warning
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")

# Setup device and seeds for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)

# Data loading and preprocessing
df = pd.read_csv('C:/Users/serem/Documents/Workspaces/Sentiment Analysis/IMDB Dataset.csv').assign(review=lambda df: df['review'].apply(lambda x: re.sub(r'<[^>]*>', '', x).lower().strip()))

# Mapping sentiment to binary
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Ensure the indices are reset
df.reset_index(drop=True, inplace=True)

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Tokenizer setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def encode_texts(tokenizer, texts):
    return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=512, return_tensors="pt")

train_encodings = encode_texts(tokenizer, train_texts)
val_encodings = encode_texts(tokenizer, val_texts)

# Load pre-trained Word2Vec model
word_vectors = api.load("word2vec-google-news-300")

# Neural network architecture
class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(150, 70),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(70, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer_stack(x).squeeze()

static_model = SentimentClassifier().to(device)

# TF-IDF vectorizer setup
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(train_texts)

# Convert texts to weighted word vectors
def texts_to_weighted_wordvecs(texts, tfidf, word_vectors):
    word2idx = vectorizer.vocabulary_
    feature_vectors = np.zeros((len(texts), word_vectors.vector_size))

    for i, text in enumerate(texts):
        words = text.split()
        weights = [tfidf[i, word2idx[word]] * word_vectors[word] for word in words if word in word2idx and word in word_vectors.key_to_index]
        feature_vectors[i] = np.sum(weights, axis=0) if weights else np.zeros(word_vectors.vector_size)

    return feature_vectors

train_wordvec_features = texts_to_weighted_wordvecs(train_texts, tfidf_matrix, word_vectors)
val_wordvec_features = texts_to_weighted_wordvecs(val_texts, tfidf_matrix, word_vectors)

# Convert features to float tensor
train_wordvec_features = torch.tensor(train_wordvec_features, dtype=torch.float32)
val_wordvec_features = torch.tensor(val_wordvec_features, dtype=torch.float32)

bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)

# Dataset and DataLoader setups
class IMDbDataset(utils.data.Dataset):
    def __init__(self, encodings=None, labels=None, embeddings=None, model_type='bert'):
        self.encodings = encodings
        self.labels = labels
        self.embeddings = embeddings
        self.model_type = model_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.model_type == 'word2vec':
            return self.embeddings[idx], self.labels.iloc[idx]
        else:
            return {key: val[idx].to(device) for key, val in self.encodings.items()}, self.labels.iloc[idx]

train_dataset_bert = IMDbDataset(encodings=train_encodings, labels=train_labels, model_type='bert')
val_dataset_bert = IMDbDataset(encodings=val_encodings, labels=val_labels, model_type='bert')
train_dataset_word2vec = IMDbDataset(embeddings=train_wordvec_features, labels=train_labels, model_type='word2vec')
val_dataset_word2vec = IMDbDataset(embeddings=val_wordvec_features, labels=val_labels, model_type='word2vec')

train_loader_word2vec = utils.data.DataLoader(train_dataset_word2vec, batch_size=16, shuffle=True)
val_loader_word2vec = utils.data.DataLoader(val_dataset_word2vec, batch_size=16, shuffle=False)
train_loader_bert = utils.data.DataLoader(train_dataset_bert, batch_size=16, shuffle=True)
val_loader_bert = utils.data.DataLoader(val_dataset_bert, batch_size=16, shuffle=False)

# Optimizers and loss function setup
bert_optimizer = optim.AdamW(bert_model.parameters(), lr=2e-5)
static_optimizer = optim.Adam(static_model.parameters(), lr=0.001)

# Initialize BCEWithLogitsLoss without passing weight, we'll handle it manually
criterion = nn.CrossEntropyLoss()

def train_and_evaluate(model, train_loader, val_loader, optimizer, device, model_type='bert', epochs=1):
    # Initialize weights for handling class imbalance
    weight_for_neg = len(df) / (2 * np.sum(df['sentiment'] == 0))
    weight_for_pos = len(df) / (2 * np.sum(df['sentiment'] == 1))

    for epoch in range(epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            optimizer.zero_grad()

            if model_type == 'bert':
                inputs = {k: v.to(device) for k, v in inputs.items()}  # Ensure data is on the correct device
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(**inputs) if model_type == 'bert' else model(inputs)

            if model_type == 'bert':
                loss = criterion(outputs.logits, labels)  # No squeezing or converting to float for BERT
            else:
                outputs = outputs.squeeze()  # Squeeze if necessary for other model types
                loss = criterion(outputs, labels.float())
                weights = torch.empty_like(labels, dtype=torch.float)
                weights[labels == 1] = weight_for_pos
                weights[labels == 0] = weight_for_neg
                loss = (loss * weights).mean()  # Apply weights if needed

            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

        model.eval()
        total_accuracy = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc="Evaluating", leave=False)):
                if model_type == 'bert':
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(device)

                outputs = model(inputs) if model_type == 'bert' else model(inputs)
                if model_type == 'bert':
                    preds = outputs.logits.argmax(dim=1)  # Use argmax for BERT model predictions
                else:
                    preds = torch.sigmoid(outputs) > 0.5  # For binary outputs from other models

                labels = labels.to(device)
                batch_accuracy = (preds == labels).float().mean().item()
                total_accuracy += batch_accuracy

    return total_accuracy / len(val_loader)

# Train and evaluate models
print("Training and evaluating Word2Vec model...")
static_val_acc = train_and_evaluate(static_model, train_loader_word2vec, val_loader_word2vec, static_optimizer, device, 'word2vec', epochs=10)
print(f"Validation Accuracy: {static_val_acc:.4f}")

print("Training and evaluating BERT model...")
bert_val_acc = train_and_evaluate(bert_model, train_loader_bert, val_loader_bert, bert_optimizer, device, 'bert', epochs=1)
print(f"Validation Accuracy: {bert_val_acc:.4f}")

# Function to get model predictions
def get_predictions(model, data_loader, device, model_type='bert'):
    model.eval()
    predictions, real_values, scores = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Predicting", leave=False)):
            if model_type == 'bert':
                inputs = {k: v.to(device) for k, v in batch[0].items()}
                outputs = model(**inputs)
                preds = torch.sigmoid(outputs.logits[:, 1])
            else:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs)

            predictions.extend(preds.cpu().numpy())
            real_values.extend(batch[1].cpu().numpy())
            scores.extend((preds > 0.5).long().cpu().numpy())

            # Debug: Print a sample of predictions and real values
            if batch_idx % 10 == 0:  # Reduce the frequency of debug prints
                print(f"Batch {batch_idx}, Sample Predictions: {preds[:5]}, Real Values: {batch[1][:5]}")

    return np.array(predictions), np.array(real_values), np.array(scores)

# Get predictions
predictions_bert, real_values_bert, scores_bert = get_predictions(bert_model, val_loader_bert, device, 'bert')
predictions_static, real_values_static, scores_static = get_predictions(static_model, val_loader_word2vec, device, 'word2vec')

# Evaluate model performance
def evaluate_performance(real_values, scores):
    accuracy = accuracy_score(real_values, scores)
    precision = precision_score(real_values, scores)
    recall = recall_score(real_values, scores)
    f1 = f1_score(real_values, scores)
    mcc = matthews_corrcoef(real_values, scores)
    return accuracy, precision, recall, f1, mcc

# Print evaluation results
accuracy_bert, precision_bert, recall_bert, f1_bert, mcc_bert = evaluate_performance(real_values_bert, scores_bert)
accuracy_static, precision_static, recall_static, f1_static, mcc_static = evaluate_performance(real_values_static, scores_static)

print(f"BERT Accuracy: {accuracy_bert:.2f}, Precision: {precision_bert:.2f}, Recall: {recall_bert:.2f}, F1 Score: {f1_bert:.2f}, MCC: {mcc_bert:.2f}")
print(f"Static Model Accuracy: {accuracy_static:.2f}, Precision: {precision_static:.2f}, Recall: {recall_static:.2f}, F1 Score: {f1_static:.2f}, MCC: {mcc_static:.2f}")

# Plotting functions
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

cm_bert = confusion_matrix(real_values_bert, scores_bert)
cm_static = confusion_matrix(real_values_static, scores_static)
plot_confusion_matrix(cm_bert, 'Confusion Matrix for BERT')
plot_confusion_matrix(cm_static, 'Confusion Matrix for Word2Vec')

# ROC Curve
def plot_roc_curve(fpr, tpr, roc_auc, title='ROC Curve'):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.fill_between(fpr, tpr, alpha=0.2, color='orange')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

fpr_bert, tpr_bert, _ = roc_curve(real_values_bert, predictions_bert)
roc_auc_bert = auc(fpr_bert, tpr_bert)
fpr_static, tpr_static, _ = roc_curve(real_values_static, predictions_static)
roc_auc_static = auc(fpr_static, tpr_static)
plot_roc_curve(fpr_bert, tpr_bert, roc_auc_bert, 'ROC Curve for BERT')
plot_roc_curve(fpr_static, tpr_static, roc_auc_static, 'ROC Curve for Word2Vec')

# Precision-Recall Curve
def plot_precision_recall_curve(recall, precision, ap, title='Precision-Recall Curve'):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'Precision-Recall curve (AP = {ap:.2f})')
    plt.fill_between(recall, precision, alpha=0.2, color='green')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

precision_bert, recall_bert, _ = precision_recall_curve(real_values_bert, predictions_bert)
ap_bert = average_precision_score(real_values_bert, predictions_bert)
precision_static, recall_static, _ = precision_recall_curve(real_values_static, predictions_static)
ap_static = average_precision_score(real_values_static, predictions_static)
plot_precision_recall_curve(recall_bert, precision_bert, ap_bert, 'Precision-Recall Curve for BERT')
plot_precision_recall_curve(recall_static, precision_static, ap_static, 'Precision-Recall Curve for Word2Vec')