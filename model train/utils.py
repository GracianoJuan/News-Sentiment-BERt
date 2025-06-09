import pandas as pd
import re
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def clean_text_advanced(text):

    if pd.isna(text):
        return ""
    
    text = str(text)
    
    text = re.sub(r'<[^>]+>', '', text)
    
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    text = re.sub(r'\S+@\S+', '', text)
    
    text = re.sub(r'[!]{3,}', '!!', text)
    text = re.sub(r'[?]{3,}', '??', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    
    return text

def balance_dataset(df, method='undersample', target_size=None):
    class_counts = df['sentiment'].value_counts()
    min_class_size = class_counts.min()
    max_class_size = class_counts.max()
    
    if method == 'undersample':
        balanced_dfs = []
        for sentiment in df['sentiment'].unique():
            class_df = df[df['sentiment'] == sentiment]
            sampled_df = class_df.sample(n=min_class_size, random_state=42)
            balanced_dfs.append(sampled_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif method == 'oversample':
        target = target_size if target_size else max_class_size
        balanced_dfs = []
        
        for sentiment in df['sentiment'].unique():
            class_df = df[df['sentiment'] == sentiment]
            current_size = len(class_df)
            
            if current_size < target:
                additional_samples = target - current_size
                oversampled = class_df.sample(n=additional_samples, replace=True, random_state=42)
                balanced_df_class = pd.concat([class_df, oversampled], ignore_index=True)
            else:
                balanced_df_class = class_df.sample(n=target, random_state=42)
            
            balanced_dfs.append(balanced_df_class)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif method == 'mixed':
        target = target_size if target_size else int(np.mean([min_class_size, max_class_size]))
        balanced_dfs = []
        
        for sentiment in df['sentiment'].unique():
            class_df = df[df['sentiment'] == sentiment]
            current_size = len(class_df)
            
            if current_size < target:
                additional_samples = target - current_size
                oversampled = class_df.sample(n=additional_samples, replace=True, random_state=42)
                balanced_df_class = pd.concat([class_df, oversampled], ignore_index=True)
            else:
                balanced_df_class = class_df.sample(n=target, random_state=42)
            
            balanced_dfs.append(balanced_df_class)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            weight_tensor = torch.tensor([self.class_weights[i] for i in range(len(self.class_weights))], 
                                       dtype=torch.float32, device=labels.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def generate_predictions(model, dataset, tokenizer, device):
    model.eval()
    predictions = []
    true_labels = []
    confidences = []

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_classes = torch.argmax(probs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            
            predictions.extend(predicted_classes.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            confidences.extend(max_probs.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels), np.array(confidences)

def predict_sentiment_enhanced(text, return_probabilities=True):
    cleaned_text = clean_text_advanced(text)
    
    if not cleaned_text:
        return "neutral", [0.33, 0.33, 0.34] if return_probabilities else "neutral"
    
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    if return_probabilities:
        prob_dict = {label: prob for label, prob in zip(label_encoder.classes_, probs.tolist()[0])}
        return predicted_label, prob_dict, confidence
    else:
        return predicted_label