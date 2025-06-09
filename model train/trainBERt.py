import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from utils import balance_dataset, NewsDataset, WeightedTrainer

# Load data
df = pd.read_csv('../data/cleaned_news_sentiment.csv')

df['description_clean'] = df['description_clean'].apply(clean_text_advanced)

df = df[df['description_clean'].str.len() > 0].reset_index(drop=True)

balance_method = 'mixed'
print(f"\nApplying {balance_method} balancing...")

df_balanced = balance_dataset(df, method=balance_method)

# Label encoding
label_encoder = LabelEncoder()
df_balanced['sentiment_encoded'] = label_encoder.fit_transform(df_balanced['sentiment'])

import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print(f"\nLabel mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i}: {label}")

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(df_balanced['sentiment_encoded']),
    y=df_balanced['sentiment_encoded']
)

class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"\nClass weights: {class_weights_dict}")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_balanced['description_clean'].tolist(), 
    df_balanced['sentiment_encoded'].tolist(), 
    test_size=0.2, 
    random_state=42,
    stratify=df_balanced['sentiment_encoded']  # Ensure balanced split
)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text_lengths = [len(tokenizer.encode(text, truncation=False)) for text in train_texts[:1000]]  # Sample for speed
avg_length = np.mean(text_lengths)
max_length = min(512, int(np.percentile(text_lengths, 95)))  # Use 95th percentile or 512, whichever is smaller

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# Load model
num_labels = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    seed=42
)

trainer = WeightedTrainer(
    class_weights=class_weights_dict,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
eval_results = trainer.evaluate()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)