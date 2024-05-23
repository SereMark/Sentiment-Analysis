import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import re
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

# Configuration Constants
EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LEN = 256  # Maximum length of tokens
MODEL_SAVE_PATH = 'Models/bert_sentiment_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MovieReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = self.reviews[item]
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(df, tokenizer):
    ds = MovieReviewDataset(
        reviews=df['review'].to_numpy(),
        targets=df['sentiment'].to_numpy(),
        tokenizer=tokenizer
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4)

def train_epoch(model, data_loader, optimizer):
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    for data in tqdm(data_loader):
        input_ids = data['input_ids'].to(DEVICE)
        attention_mask = data['attention_mask'].to(DEVICE)
        targets = data['targets'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += (preds == targets).sum()
        total_predictions += targets.size(0)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.double() / total_predictions, np.mean(losses)

def eval_model(model, data_loader):
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(DEVICE)
            attention_mask = data['attention_mask'].to(DEVICE)
            targets = data['targets'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += (preds == targets).sum()
            total_predictions += targets.size(0)
            losses.append(loss.item())

    return correct_predictions.double() / total_predictions, np.mean(losses)

def remove_html_tags(text):
    return re.sub('<.*?>', '', text)

def main():
    print("Verifying CUDA availability...")
    print(f"Device: {DEVICE}")

    # Load and preprocess data
    df = pd.read_csv("Data/IMDB Dataset.csv")
    df['review'] = df['review'].apply(remove_html_tags)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(DEVICE)

    train_data_loader = create_data_loader(df_train, tokenizer)
    test_data_loader = create_data_loader(df_test, tokenizer)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train_acc, train_loss = train_epoch(model, train_data_loader, optimizer)
        print(f'Train loss {train_loss}, accuracy {train_acc}')

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    test_acc, test_loss = eval_model(model, test_data_loader)
    print(f'Test loss {test_loss}, accuracy {test_acc}')

if __name__ == "__main__":
    main()