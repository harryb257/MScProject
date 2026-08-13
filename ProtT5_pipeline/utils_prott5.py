import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    set_seed
)

from peft import LoraConfig, get_peft_model

from transformers import T5EncoderModel, T5Tokenizer

import pandas as pd
import numpy as np
import random

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    roc_auc_score)

from pathlib import PurePosixPath


# Set seeds for reproducibility
def set_seeds(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    # Hugging face
    set_seed(s)
    # Cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Classification head
class PerResidueClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


class ProtT5ForResidueClassification(nn.Module):
    def __init__(self, checkpoint, mode, embedding_dim):
        super().__init__()

        # Prott5 encoder
        self.encoder = T5EncoderModel.from_pretrained(checkpoint)

        if mode == 'full':
            # Add classification head using existing 2 layer nn classifier
            self.classifier = PerResidueClassifier(embedding_dim)

        elif mode == 'LoRA':
            # Lora Config as per Nature paper
            peft_config = LoraConfig(r=4, lora_alpha=1, bias="all", target_modules=["q","k","v","o"])
            # Inject LoRA adapters
            self.encoder = get_peft_model(self.encoder, peft_config)

            # Add classification head using existing 2 layer nn classifier
            self.classifier = PerResidueClassifier(embedding_dim)

    # Forward pass through the ProtT5 encoder and classifier to generate embeddings and losses respectively
    def forward(
        self,
        input_ids,
        attention_mask,
        labels):

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state

        logits = self.classifier(embeddings)
        loss_fcn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fcn(logits.reshape(-1, 2), labels.reshape(-1))

        return {
            "loss": loss,
            "logits": logits,
        }


# load prott5 models for fine tuning
def load_prott5_with_classification_head(checkpoint, mode, embedding_dim):

    if mode == 'full':
        model = ProtT5ForResidueClassification(checkpoint, mode, embedding_dim)
        tokenizer = T5Tokenizer.from_pretrained(checkpoint, do_lower_case=False)
        return model, tokenizer

    else:
        model = ProtT5ForResidueClassification(checkpoint, mode, embedding_dim)
        tokenizer = T5Tokenizer.from_pretrained(checkpoint, do_lower_case=False)
        return model, tokenizer


def select_datasets(files):
    # Define higher, lower and test data
    lower_level = None
    higher_level = None
    target = None

    for file in files:
        name = PurePosixPath(file).name

        if 'Lower' in name:
            lower_level = "s3://" + file
        elif 'Higher' in name:
            higher_level = "s3://" + file
        elif 'Target' in name:
            target = "s3://" + file

    return lower_level, higher_level, target


def sliding_window(df, window_size=1024, stride=512):
    """
    Function to create 1024 length splits for proteins >1024 in length, using stride length of 512
    Inserts new splits into a datatable with split suffix
    """
    rows = []

    for _, row in df.iterrows():
        seq = row['sequence']
        seq_len = len(seq)

        if seq_len <= window_size:
            rows.append(row.copy())
            continue

        start = 0
        split = 1
        while start < seq_len:
            # End slicing variable
            end = start + window_size

            # Create copy of the row
            new_row = row.copy()
            # Slice amino acid sequence by start end index
            new_row['sequence'] = seq[start:end]

            # Slice columns containing lists
            for col in ['label', 'position']:
                new_row[col] = row[col][start:end]

            # Add split suffix to Info_protein_id
            new_row['Info_protein_id'] = str(new_row['Info_protein_id']) + '_' + str(split)
            # Append window to row list
            rows.append(new_row)

            # Stop after the final window
            if end >= seq_len:
                break

            start += stride
            split += 1

    return pd.DataFrame(rows).reset_index(drop=True)


def delete_unlabelled_rows(df):
    """ Function to delete rows with all unlabelled residues """
    return df[df['label'].apply(lambda labels: 0 in labels or 1 in labels)].reset_index(drop=True)


def preprocess_csv(csv_file, return_origin_df_len=False):
    """ Function to preprocess a csv file.
    Imports csv, refactors label column and masks NA values, aggregates into wide format
    applies sliding window and deletes unlabelled rows
    """
    preprocessed_df = pd.read_csv(csv_file)

    # Drop rows where 'Info_split' == 'NA'
    preprocessed_df = preprocessed_df[preprocessed_df['Info_split'] != 'NA']

    # Mask n/a values with -100
    preprocessed_df['Class'] = preprocessed_df['Class'].fillna(-100).astype('int32')
    preprocessed_df['Class'] = preprocessed_df['Class'].replace(-1, 0)

    # Sort values before aggregation
    preprocessed_df = preprocessed_df.sort_values(['Info_protein_id', 'Info_pos'])

    # Aggregate columns for wide format
    preprocessed_df = preprocessed_df.groupby(['Info_protein_id', 'Info_group', 'Info_split'], as_index=False).agg(
        sequence=('Info_AA', ' '.join),
        label=('Class', list),
        position=('Info_pos', list))

    # Apply sliding window
    preprocessed_df = sliding_window(preprocessed_df)

    # Delete unlabelled rows
    preprocessed_df = delete_unlabelled_rows(preprocessed_df)

    return preprocessed_df


class SequenceDataset(torch.utils.data.Dataset):
    # Create custom dataset class
    def __init__(self, df):
        self.protein_id = df['Info_protein_id']
        self.sequence = df['sequence']
        self.position = df['position']
        self.labels = df['label']

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return {
            'protein_id': self.protein_id.iloc[idx],
            'sequence': self.sequence.iloc[idx],
            'position': self.position.iloc[idx],
            'label': self.labels.iloc[idx]
        }


def collate_fn(batch):
    """
    Function to create a custom collator to maintain length of items within the batch
    """
    return {
        'protein_id': [x['protein_id'] for x in batch],
        'sequence': [x['sequence'] for x in batch],
        'position': [x['position'] for x in batch],
        'label': [x['label'] for x in batch]
    }

def batch_create(dataset, batch_size, tokenizer, model, mode):
    """
    Function to create a DataLoader instance using the custom collate function
    """
    # Create a DataLoader instance using the cv dataset and the custom collate function
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    emb_output_list = []

    model.eval()

    # Loop through each batch of tensors and apply tokenisation to each sequence
    for batch in loader:

        inputs = tokenizer(
            batch['sequence'],
            padding=True,
            truncation=False,
            return_tensors='pt'
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Freeze model weights and calculate embeddings
        with torch.inference_mode():

            # Use the loaded model's T5 encoder to generate embeddings
            outputs = model.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'])
            embeddings = outputs.last_hidden_state[
                :, :-1, :].cpu()  # Slice embeddings to remove end CLS/EOS tokens

        # Convert labels to tensors
        labels = [torch.tensor(x) for x in batch['label']]

        # Pad each label to the size of the largest embedding within the batch
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        assert embeddings.shape[1] == labels.shape[1]

        # Append the embeddings, label and masks per batch to an output list
        emb_output_list.append({
            'embeddings': embeddings,
            'labels': labels
        })

    return emb_output_list


def create_datasets_for_clf(train_dict, val_dict, batch_size, tokenizer, model, mode):
    """
    Function to create datasets and store in dictionaries for cross validation
    """

    train_datasets = {
        key: SequenceDataset(value)
        for key, value in train_dict.items()
    }

    val_datasets = {
        key: SequenceDataset(value)
        for key, value in val_dict.items()
    }

    train_loaded = {}
    for key, value in train_datasets.items():
        train_loaded[key] = batch_create(value, batch_size, tokenizer, model, mode)

    val_loaded = {}
    for key, value in val_datasets.items():
        val_loaded[key] = batch_create(value, batch_size, tokenizer, model, mode)

    return train_loaded, val_loaded



def preprocess_higher_level(tokenizer, data):
    """
    Preprocess each df row to tokenise the sequences and pad labels to match max length
    """
    inputs = tokenizer(
        data['sequence'],
        padding='max_length',
        truncation=True,
        max_length=1024,
    )

    labels = data['label']

    print('inputs', inputs)

    # True per-residue labels, then -100 for cls + all PAD
    labels =  labels + [-100] * (1024 - len(labels))
    print('labels', labels)

    return {'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels}


def trainable_parameters_summary(model):
    # Save a table of the number and percent of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in model.parameters())

    trainable_percent = trainable_params / total_params * 100

    parameter_summary = {
        'trainable_params': trainable_params,
        'total_params': total_params,
        'trainable_percent': trainable_percent}

    parameter_summary = pd.DataFrame([parameter_summary])

    return parameter_summary


# Define Trainer parameters
def compute_metrics(p):
    # Separate logits and labels
    pred, labels = p

    # Return index of higher position (neg or positive residue) per row
    max_pred = np.argmax(pred, axis=-1)

    # Calculate probabilities to be used for AUC
    probs = torch.softmax(torch.tensor(pred), dim=-1)
    probs = probs[:, :, 1]

    # Create mask for unlabelled positions
    mask = labels != -100

    # Calculate metrics
    accuracy = accuracy_score(y_true=labels[mask], y_pred=max_pred[mask])
    recall = recall_score(y_true=labels[mask], y_pred=max_pred[mask])
    precision = precision_score(y_true=labels[mask], y_pred=max_pred[mask])
    f1 = f1_score(y_true=labels[mask], y_pred=max_pred[mask])
    mcc = matthews_corrcoef(y_true=labels[mask], y_pred=max_pred[mask])
    auc = roc_auc_score(y_true=labels[mask], y_score=probs[mask])

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'mcc': mcc, 'auc': auc}


def class_weighting_for_clf(lower_stacked):
    # Convert to csv
    df_lower_stacked = pd.read_csv(lower_stacked)

    # Drop rows where 'Info_split' == 'NA'
    df_lower_stacked = df_lower_stacked[df_lower_stacked['Info_split'] != 'NA']

    # Determine frequency of positive vs negative labels to be used to weight the loss function
    neg_class = len(df_lower_stacked[df_lower_stacked['Class'] == -1])
    pos_class = len(df_lower_stacked[df_lower_stacked['Class'] == 1])

    # Total labelled
    total = neg_class + pos_class

    # Set the weight as the inverse proportion of the tota
    neg_weight = torch.tensor([total / (2 * neg_class)])
    pos_weight = torch.tensor([total / (2 * pos_class)])

    # Concat tensors
    weight = torch.cat([neg_weight, pos_weight])

    return weight


def train_one_epoch(model, train_dataloader, optimiser, loss_function):
    # Put the clf in training mode
    model.train()

    running_loss = 0

    train_preds = []
    train_labels = []

    for batch in train_dataloader:
        inputs = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)

        # Zero model gradients per batch
        optimiser.zero_grad()

        # Caluclate logits by passing embeddings through the classifier
        outputs = model(inputs)

        # Compute loss and gradients
        loss = loss_function(outputs.reshape(-1, 2), labels.reshape(-1))

        # Calculate the gradients through the network
        loss.backward()

        # Adjust learning weights
        optimiser.step()

        # Add the loss to a running counter
        running_loss += loss.item()

        # Predicted labels, assign pos or negative based on argmax of the pos / neg class logits, flatten
        train_preds.append(torch.argmax(outputs, dim=-1).reshape(-1))
        train_labels.append(labels.reshape(-1))

    # Concat the batched tensors to lists
    train_preds = torch.cat(train_preds)
    train_labels = torch.cat(train_labels)

    # Create mask for calculating metrics only on labelled residues
    mask = train_labels != -100

    train_preds = train_preds[mask].cpu().numpy()
    train_labels = train_labels[mask].cpu().numpy()

    metrics = {
        'loss': running_loss / len(train_dataloader),
        'accuracy': accuracy_score(
            train_labels,
            train_preds)
    }

    return metrics


def val_one_epoch(model, val_dataloader, loss_function):
    # Put the classifier into evaluation mode
    model.eval()

    running_loss = 0

    val_preds = []
    val_labels = []
    val_probs = []

    # Training loop
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)

            # Calculate logits by passing embeddings through the classifier
            outputs = model(inputs)

            # Compute loss and gradients
            loss = loss_function(outputs.reshape(-1, 2), labels.reshape(-1))

            # Add the loss to a running counter
            running_loss += loss.item()

            # Predicted labels, assign pos or negative based on argmax of the pos / neg class logits, flatten and append
            val_preds.append(torch.argmax(outputs, dim=-1).reshape(-1))
            val_labels.append(labels.reshape(-1))

            # Calculate probabilities to be used for AUC
            probs = torch.softmax(outputs, dim=-1)
            pos_probs_flat = probs[:, :, 1].reshape(-1)
            val_probs.append(pos_probs_flat)

    # Concat the batched tensors in the lists
    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)
    val_probs = torch.cat(val_probs)

    mask = val_labels != -100

    # Apply mask while still tensors
    val_preds = val_preds[mask].cpu().numpy()
    val_labels = val_labels[mask].cpu().numpy()
    val_probs = val_probs[mask].cpu().numpy()

    # Calculate validation metrics
    metrics = {
        'loss': running_loss / len(val_dataloader),
        'accuracy': accuracy_score(val_labels, val_preds),
        'precision': precision_score(val_labels, val_preds),
        'recall': recall_score(val_labels, val_preds),
        'f1': f1_score(val_labels, val_preds),
        'mcc': matthews_corrcoef(val_labels, val_preds),
        'auc': roc_auc_score(val_labels, val_probs)
    }

    return metrics, val_labels, val_probs, val_preds


def train_classifier(embedding_dim,
                     train_dataloader,
                     val_dataloader,
                     loss_function,
                     epochs=20):
    logs = []
    fold_labels_probs_preds_epoch = {}

    for key in train_dataloader.keys():

        model = PerResidueClassifier(embedding_dim).to(device)

        optimiser = optim.AdamW(model.parameters(), lr=0.001)

        best_auc = 0

        for epoch in range(epochs):
            train_metrics = train_one_epoch(model, train_dataloader[key], optimiser, loss_function)

            val_metrics, val_labels, val_probs, val_preds = val_one_epoch(model, val_dataloader[key], loss_function)

            logs.append({
                'fold': key,
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'val_mcc': val_metrics['mcc'],
                'val_auc': val_metrics['auc'],
            })

            fold_labels_probs_preds_epoch[(key, epoch+1)] = (val_labels, val_probs, val_preds)

    history = pd.DataFrame(logs)

    return history, fold_labels_probs_preds_epoch


def train_final_classifier(model,
                           train_dataloader,
                           loss_function,
                           epochs=20):
    logs = []

    model = model.to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_dataloader, optimiser, loss_function)

        logs.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy']
        })

    history = pd.DataFrame(logs)

    return {
        "history": history,
        "model_state_dict": model.state_dict(),
        "epochs": epochs,
    }
