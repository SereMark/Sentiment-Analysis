import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from torch.nn import functional as F
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration
class Config:
    EPOCHS = 4
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    MAX_LEN = 256  # Maximum length of tokens
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH = 'Models/bert_sentiment_model.pth'

# Custom dataset class
class MovieReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

# Data loader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = MovieReviewDataset(
        reviews=df.review.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

# Model setup
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    model.to(device)
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )

        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    model.to(device)
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )

            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def main():
    print("Verifying CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    print("Loading data...")
    # Load data
    df = pd.read_csv("Data/IMDB Dataset.csv")
    df['review'] = df['review'].apply(remove_html_tags)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Training data size: {len(df_train)}, Test data size: {len(df_test)}")

    print("Initializing tokenizer and model...")
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Data loaders
    print("Creating data loaders...")
    train_data_loader = create_data_loader(df_train, tokenizer, Config.MAX_LEN, Config.BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, Config.MAX_LEN, Config.BATCH_SIZE)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    print(f"Using device: {Config.DEVICE}")

    # Training loop
    for epoch in range(Config.EPOCHS):
        print(f'Epoch {epoch + 1}/{Config.EPOCHS}')
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            F.cross_entropy,
            optimizer,
            Config.DEVICE,
            len(df_train)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')

    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # Evaluate model
    print("Evaluating model...")
    test_acc, test_loss = eval_model(
        model,
        test_data_loader,
        F.cross_entropy,
        Config.DEVICE,
        len(df_test)
    )
    print(f'Test loss {test_loss} accuracy {test_acc}')

if __name__ == "__main__":
    main()