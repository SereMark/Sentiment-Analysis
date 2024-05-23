import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from torch.nn import functional as F
from tqdm import tqdm

# Configuration
class Config:
    EPOCHS = 4
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    MAX_LEN = 256  # Maximum length of tokens
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# Main function
def main():
    # Load data
    df = pd.read_csv("Data/IMDB Dataset.csv")
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Data loaders
    train_data_loader = create_data_loader(df_train, tokenizer, Config.MAX_LEN, Config.BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, Config.MAX_LEN, Config.BATCH_SIZE)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

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

if __name__ == "__main__":
    main()
