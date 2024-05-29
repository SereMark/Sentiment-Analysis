# Sentiment Analysis of IMDb Reviews

# This project employs sentiment analysis to classify IMDb movie reviews using static embeddings (GloVe) and contextual embeddings (BERT). 
# This comparison aims to evaluate their effectiveness in sentiment classification, providing insights that could guide model selection 
# and application strategies in natural language processing tasks.

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
from tqdm import tqdm
import warnings

# Suppress specific warnings from torch to maintain a clean output, especially those not relevant to the execution such as 
# compilation details and deprecation warnings which do not affect the model's performance.
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")

# Set the computation device to GPU if available for faster processing, otherwise use CPU. This choice is crucial as BERT and other deep learning
# models are computationally intensive and can benefit significantly from GPU acceleration.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the dataset: Converting all reviews to lowercase to ensure uniformity and mapping sentiments to binary values for classification.
imdb_data = pd.read_csv('IMDB Dataset.csv')
imdb_data['review'] = imdb_data['review'].str.lower()  # Lowercasing to normalize text.
imdb_data['sentiment'] = imdb_data['sentiment'].map({'positive': 1, 'negative': 0})  # Mapping textual sentiment labels to binary for processing.

# Split the dataset into training and validation sets with an 80/20 split. This helps in validating the model on unseen data.
# Random state is fixed at 42 to ensure reproducibility of the split.
train_reviews, val_reviews, train_sentiments, val_sentiments = train_test_split(
    imdb_data['review'].tolist(), imdb_data['sentiment'].tolist(), test_size=0.2, random_state=42
)

# Separate reviews based on sentiment for possible exploratory data analysis, such as understanding the distribution of word counts in positive vs. negative reviews.
positive_reviews = [len(review.split()) for review, sentiment in zip(imdb_data['review'].tolist(), imdb_data['sentiment'].tolist()) if sentiment == 1]
negative_reviews = [len(review.split()) for review, sentiment in zip(imdb_data['review'].tolist(), imdb_data['sentiment'].tolist()) if sentiment == 0]

# Plot histogram of review lengths for each sentiment: This visualization helps to understand if the length of reviews
# correlates with sentiment, potentially influencing model training and feature engineering.
plt.figure()
plt.hist(positive_reviews, bins=30, color='blue', alpha=0.7, label='Positive')
plt.hist(negative_reviews, bins=30, color='red', alpha=0.7, label='Negative')
plt.title('Histogram of Review Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

# Initialize GloVe embeddings: GloVe is used here as a static word embedding method to represent words in a lower-dimensional 
# vector space. This model is pretrained and captures semantic meaning based on word co-occurrence in a corpus.
glove_embeddings = vocab.GloVe(name='6B', dim=100)  # Using a 100-dimensional space to represent each word.

# Function to fetch GloVe embeddings for a text. This function tokenizes the text, retrieves embeddings for each token if available, 
# and averages them to create a fixed-length input vector for the neural network. This average embedding serves as a simple but effective 
# way to encode text information.
def fetch_glove_embeddings(text):
    tokens = text.split()
    embeddings = [glove_embeddings[t].unsqueeze(0) for t in tokens if t in glove_embeddings.stoi]
    return torch.cat(embeddings, dim=0).mean(dim=0) if embeddings else torch.zeros(glove_embeddings.dim)

train_glove_embeddings = [fetch_glove_embeddings(text) for text in train_reviews]
val_glove_embeddings = [fetch_glove_embeddings(text) for text in val_reviews]

# Dataset class for GloVe embeddings: This custom dataset prepares batches from the precomputed GloVe embeddings, which will be used
# to train a simple neural network classifier. It encapsulates the data handling for PyTorch model training.
class GloVeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.stack(embeddings)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {'embedding': self.embeddings[idx].clone().detach(), 'label': self.labels[idx]}

# BERT tokenizer and model setup: BERT requires a specific type of text preprocessing with a tokenizer that splits the text into tokens 
# and maps them to their corresponding indices in the pretrained model's vocabulary. 
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset class for BERT tokenized data: Similar to the GloVe dataset but tailored for BERT inputs. It processes text through the BERT tokenizer,
# handling max length and padding automatically, and prepares input tensors for BERT model processing.
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

# Generalized model class for sentiment analysis: This class can switch between a BERT-based classifier and a simpler neural network 
# depending on the 'model_type' parameter. It demonstrates polymorphism in model design within PyTorch.
class SentimentClassifier(nn.Module):
    def __init__(self, model_type='BERT'):
        super().__init__()
        if model_type == 'BERT':
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.dropout = nn.Dropout(0.3)  # Dropout for regularization.
            self.classifier = nn.Linear(self.model.config.hidden_size, 2)  # The final output layer that maps BERT outputs to two classes.
        else:
            self.fc1 = nn.Linear(100, 50)  # First layer of a simple neural network for GloVe embeddings.
            self.relu = nn.ReLU()  # ReLU activation function to introduce non-linearity.
            self.dropout = nn.Dropout(0.5)  # Higher dropout rate for simpler models to prevent overfitting.
            self.fc2 = nn.Linear(50, 2)  # Final layer outputting to two classes.

    def forward(self, x, attention_mask=None):
        if hasattr(self, 'model'):
            output = self.model(x, attention_mask=attention_mask)[1]  # Processes input through BERT.
            output = self.dropout(output)
            return self.classifier(output)  # Applies classifier to BERT's output.
        else:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            return self.fc2(x)  # Processes input through simple neural network.

# Load data into DataLoader: DataLoaders are a key component in PyTorch for handling batches of data. They automate the process of loading
# and shuffling data, providing an iterable over the dataset for training and evaluation phases.
train_loader = DataLoader(BertDataset(train_reviews, train_sentiments), batch_size=16, shuffle=True)
val_loader = DataLoader(BertDataset(val_reviews, val_sentiments), batch_size=16, shuffle=False)
glove_train_loader = DataLoader(GloVeDataset(train_glove_embeddings, train_sentiments), batch_size=16, shuffle=True)
glove_val_loader = DataLoader(GloVeDataset(val_glove_embeddings, val_sentiments), batch_size=16, shuffle=False)

# Initialize models, optimizers, and loss function: The choice of optimizer and its settings (like learning rate) can greatly affect
# the training dynamics and final performance of the models. Adam optimizer is used here due to its effectiveness in handling sparse gradients and adaptive learning rate capabilities.
bert_classifier = SentimentClassifier('BERT').to(device)  # BERT-based model initialized and moved to GPU if available.
glove_classifier = SentimentClassifier('GloVe').to(device)  # GloVe-based simple neural network model also moved to GPU.
optimizer_bert = torch.optim.Adam(bert_classifier.parameters(), lr=2e-5)  # Learning rate for BERT should be very low to prevent overfitting.
optimizer_glove = torch.optim.Adam(glove_classifier.parameters(), lr=5e-4)  # Higher learning rate for simpler model.
loss_function = nn.CrossEntropyLoss()  # CrossEntropyLoss is suitable for binary classification tasks.

# Training function: This function encapsulates the training loop, which iterates over the dataset multiple times (epochs), processing each batch,
# computing loss, and updating model weights. It prints out the training progress and loss at each epoch, providing insight into how well the model is learning.
def train(model, loader, optimizer, loss_function, device, model_type='BERT', epochs=1):
    model.train()  # Sets the model in training mode.
    all_epoch_losses = []  # List to store loss per epoch for later analysis.
    print("Starting training...")
    for epoch in range(epochs):  # Loop over the dataset multiple times.
        epoch_loss = 0
        print(f"Epoch {epoch+1}/{epochs} started...")
        for batch in tqdm(loader, desc=f"Training Epoch {epoch+1}"):  # tqdm is used to show a progress bar for each epoch.
            optimizer.zero_grad()  # Clears old gradients from the last step.
            if model_type == 'BERT':
                input_ids = batch['input_ids'].to(device)  # Loads batch data onto the GPU.
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)  # Passes batch through the model.
            else:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].to(device)
                outputs = model(embeddings)
            loss = loss_function(outputs, labels)  # Computes loss between model outputs and actual labels.
            loss.backward()  # Backpropagation step to compute gradients.
            optimizer.step()  # Adjusts weights based on gradients.
            epoch_loss += loss.item()  # Sum up the loss for reporting.
        avg_loss = epoch_loss / len(loader)  # Average loss for the epoch.
        all_epoch_losses.append(avg_loss)  # Store the average loss.
        print(f"Epoch {epoch+1} completed. Training Loss: {avg_loss}")
    print("Training completed.")
    return all_epoch_losses  # Returns the list of average losses for each epoch for analysis.

# Train both models: The models are trained for 10 epochs each. The number of epochs is a parameter that can be adjusted based on the model's performance and training time.
bert_losses = train(bert_classifier, train_loader, optimizer_bert, loss_function, device, 'BERT', epochs=10)
glove_losses = train(glove_classifier, glove_train_loader, optimizer_glove, loss_function, device, 'GloVe', epochs=10)

# Evaluation function: This function evaluates the trained model on a validation set. It disables gradient calculations, saving memory and speeding up computation,
# essential for evaluating large models like BERT. It computes the model's predictions, compares them to the true labels, and calculates probabilities of positive sentiment,
# which are useful for ROC and precision-recall curves analysis.
def evaluate(model, loader, device, model_type='BERT'):
    model.eval()  # Sets the model to evaluation mode.
    predictions, true_labels, probabilities = [], [], []
    print("Starting evaluation...")
    with torch.no_grad():  # Disables gradient calculation to save memory and speeds up computation.
        for batch in tqdm(loader, desc="Evaluating"):  # Iterates over each batch in the loader.
            if model_type == 'BERT':
                input_ids = batch['input_ids'].to(device)  # Load batch data onto the GPU.
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)  # Pass the data through the model.
                outputs = outputs.softmax(dim=-1)  # Apply softmax to get probabilities.
            else:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].to(device)
                outputs = model(embeddings)
                outputs = outputs.softmax(dim=-1)
            _, preds = torch.max(outputs, dim=1)  # Get the predicted labels.
            predictions.extend(preds.cpu().numpy())  # Store predictions to compute metrics later.
            true_labels.extend(labels.cpu().numpy())  # Store true labels for metrics calculation.
            probabilities.extend(outputs.cpu().numpy()[:, 1])  # Store probabilities for ROC and precision-recall analysis.
    print("Evaluation completed.")
    print(classification_report(true_labels, predictions))  # Prints classification metrics to evaluate model performance.
    return predictions, true_labels, probabilities  # Returns predictions, true labels, and probabilities for further analysis.

# Evaluate both models: After training, both models are evaluated on the validation set to compare their performance objectively.
bert_preds, bert_labels, bert_probs = evaluate(bert_classifier, val_loader, device, 'BERT')
glove_preds, glove_labels, glove_probs = evaluate(glove_classifier, glove_val_loader, device, 'GloVe')

# Training loss comparison function: This function plots the training losses of both models across epochs to visualize and compare
# how each model learns over time. A lower or smoother loss curve can indicate a better learning process.
def plot_combined_training_loss(losses1, losses2, name1, name2):
    plt.figure()
    plt.plot(losses1, label=f'Training Loss for {name1}')  # Plot BERT training losses.
    plt.plot(losses2, label=f'Training Loss for {name2}')  # Plot GloVe training losses.
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

# Plot training loss comparison: After training, the training loss curves of both models are plotted to compare their learning behaviors.
plot_combined_training_loss(bert_losses, glove_losses, 'BERT', 'GloVe')

# Histogram of predicted probabilities comparison function: This function plots histograms of the predicted probabilities for positive sentiment
# from both models. This visualization helps understand the confidence of the models in their predictions and can reveal how each model discriminates
# between classes.
def plot_combined_probability_histogram(probs1, probs2, name1, name2):
    plt.figure()
    plt.hist(probs1, bins=10, alpha=0.75, label=f'Predicted Probabilities {name1}')  # Histogram for BERT probabilities.
    plt.hist(probs2, bins=10, alpha=0.75, label=f'Predicted Probabilities {name2}')  # Histogram for GloVe probabilities.
    plt.title('Histogram of Predicted Probabilities')
    plt.xlabel('Probability of Positive Sentiment')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Plot histogram of predicted probabilities: This plot provides insights into how certain or uncertain the models are in their predictions,
# which can be crucial for applications where confidence thresholds are important.
plot_combined_probability_histogram(bert_probs, glove_probs, 'BERT', 'GloVe')

# ROC curve comparison function: This function plots the Receiver Operating Characteristic (ROC) curves for both models. ROC curves are useful for evaluating
# the diagnostic ability of binary classifiers, which is essential in tasks like sentiment analysis. The area under the curve (AUC) can be used as a summary
# of model performance, with an AUC close to 1.0 indicating a very good model and 0.5 a model with no discriminative ability.
def plot_combined_roc_curve(true_labels1, probs1, true_labels2, probs2, name1, name2):
    fpr1, tpr1, _ = roc_curve(true_labels1, probs1)  # Calculate false positive rate and true positive rate for BERT.
    roc_auc1 = auc(fpr1, tpr1)  # Calculate area under the ROC curve for BERT.
    fpr2, tpr2, _ = roc_curve(true_labels2, probs2)  # Calculate false positive rate and true positive rate for GloVe.
    roc_auc2 = auc(fpr2, tpr2)  # Calculate area under the ROC curve for GloVe.
    
    plt.figure()
    plt.plot(fpr1, tpr1, label=f'{name1} ROC curve (area = {roc_auc1:.2f})')  # Plot BERT ROC curve.
    plt.plot(fpr2, tpr2, label=f'{name2} ROC curve (area = {roc_auc2:.2f})')  # Plot GloVe ROC curve.
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')  # Plot a diagonal line for reference.
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curve comparison: This comparison allows us to see which model better discriminates between positive and negative sentiments,
# as indicated by the area under the curve.
plot_combined_roc_curve(bert_labels, bert_probs, glove_labels, glove_probs, 'BERT', 'GloVe')

# Confusion matrix comparison function: This function plots the confusion matrices for both models. A confusion matrix is a table that is often used
# to describe the performance of a classification model on a set of test data for which the true values are known. It allows the visualization of the
# performance of an algorithm.
def plot_combined_confusion_matrix(true_labels1, predictions1, true_labels2, predictions2, name1, name2):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure and a set of subplots.

    cf_matrix1 = confusion_matrix(true_labels1, predictions1)  # Compute confusion matrix for BERT.
    sns.heatmap(cf_matrix1, annot=True, fmt='g', ax=ax[0])  # Plot heatmap for BERT confusion matrix.
    ax[0].set_title(f'Confusion Matrix for {name1}')  # Set title for BERT confusion matrix.
    ax[0].set_xlabel('Predicted')  # Set x-axis label for BERT confusion matrix.
    ax[0].set_ylabel('Actual')  # Set y-axis label for BERT confusion matrix.
    
    cf_matrix2 = confusion_matrix(true_labels2, predictions2)  # Compute confusion matrix for GloVe.
    sns.heatmap(cf_matrix2, annot=True, fmt='g', ax=ax[1])  # Plot heatmap for GloVe confusion matrix.
    ax[1].set_title(f'Confusion Matrix for {name2}')  # Set title for GloVe confusion matrix.
    ax[1].set_xlabel('Predicted')  # Set x-axis label for GloVe confusion matrix.
    ax[1].set_ylabel('Actual')  # Set y-axis label for GloVe confusion matrix.
    
    plt.show()

# Plot confusion matrix comparison: This plot helps to visualize where the models perform well and where they make errors,
# providing insights into possible improvements or adjustments in model training or preprocessing.
plot_combined_confusion_matrix(bert_labels, bert_preds, glove_labels, glove_preds, 'BERT', 'GloVe')

# Precision-Recall curve comparison function: This function plots the precision-recall curves for both models. Precision-Recall is a useful measure of
# success of prediction when the classes are very imbalanced. High area under the curve represents both high recall and high precision, where high precision
# relates to a low false positive rate, and high recall relates to a low false negative rate.
def plot_combined_precision_recall_curve(true_labels1, probs1, true_labels2, probs2, name1, name2):
    precision1, recall1, _ = precision_recall_curve(true_labels1, probs1)  # Calculate precision and recall for BERT.
    precision2, recall2, _ = precision_recall_curve(true_labels2, probs2)  # Calculate precision and recall for GloVe.
    
    plt.figure()
    plt.plot(recall1, precision1, label=f'{name1} Precision-Recall curve')  # Plot BERT precision-recall curve.
    plt.plot(recall2, precision2, label=f'{name2} Precision-Recall curve')  # Plot GloVe precision-recall curve.
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend()
    plt.show()

# Plot precision-recall curve comparison: This plot provides insights into the trade-off between recall and precision for both models,
# helping to understand which model performs better in maintaining a balance between precision and recall.
plot_combined_precision_recall_curve(bert_labels, bert_probs, glove_labels, glove_probs, 'BERT', 'GloVe')

# Sentiment prediction function: This function takes a text input and predicts the sentiment using the specified model. It handles both BERT and GloVe models,
# adjusting the processing according to the model type. This function is crucial for deploying the trained models in a production environment or for testing them on new data.
def predict_sentiment(text, model, tokenizer, device):
    if not text.strip():  # Check if the text is not just empty or spaces.
        return "No input", 0  # Return no input if the text is empty.
    model.eval()  # Ensure the model is in evaluation mode, which turns off dropout and batch normalization.
    text = text.lower()  # Lowercase the text to maintain consistency with training data preprocessing.
    try:
        if tokenizer:
            tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors="pt").to(device)  # Tokenize the text and move to GPU if available.
            with torch.no_grad():  # Disable gradient calculation.
                output = model(tokenized['input_ids'], tokenized['attention_mask']).softmax(dim=-1)  # Get model predictions and apply softmax to convert to probabilities.
        else:
            embeddings = fetch_glove_embeddings(text).unsqueeze(0).to(device)  # Fetch GloVe embeddings and move to GPU if available.
            with torch.no_grad():  # Disable gradient calculation.
                output = model(embeddings).softmax(dim=-1)  # Get model predictions and apply softmax to convert to probabilities.
        pred = torch.argmax(output, dim=1).item()  # Get the predicted class (0 or 1).
        certainty = output[0, pred].item()  # Get the probability associated with the predicted class.
        sentiment = 'Positive' if pred == 1 else 'Negative'  # Convert the predicted class to a sentiment label.
        return sentiment, certainty * 100  # Return the sentiment and the certainty as a percentage.
    except Exception as e:
        return f"Error: {str(e)}", 0  # Return the error message if an exception occurs.

# Sentiment prediction widget for google colab: This widget allows users to interactively enter text and get sentiment predictions.
# It updates the predictions every time the text changes, providing an interactive way to test the models.
def update_output(change):
    text_input = change.new  # Get the new text from the widget.
    if text_input:  # Update only if the text is not empty.
        bert_sentiment, bert_certainty = predict_sentiment(text_input, bert_classifier, bert_tokenizer, device)  # Predict sentiment using BERT.
        glove_sentiment, glove_certainty = predict_sentiment(text_input, glove_classifier, None, device)  # Predict sentiment using GloVe.
        with output:
            clear_output(wait=True)  # Clear the previous output.
            print(f"BERT: Sentiment - {bert_sentiment}, Certainty - {bert_certainty:.2f}%")  # Display BERT predictions.
            print(f"GloVe: Sentiment - {glove_sentiment}, Certainty - {glove_certainty:.2f}%")  # Display GloVe predictions.

# Widget setup: This section sets up the interactive widget for the sentiment prediction. It includes a text input field and an output area
# where predictions are displayed.
text_input = widgets.Text(placeholder='Type something here...', description='Input:', disabled=False)  # Create a text input widget.
output = widgets.Output()  # Create an output widget to display predictions.
text_input.observe(update_output, names='value')  # Set up an observer to call update_output when the text changes.
display(text_input, output)  # Display the widgets.