import Pfeature
import torch
import pandas as pd
import numpy as np
import random
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    roc_auc_score)

from pathlib import Path
from pathlib import PurePosixPath

import os
import tempfile
from tqdm import tqdm

from Pfeature.pfeature import aac_wp, atc_wp, btc_wp, pcp_wp, sep_wp, ctc_wp

# Set device to Apple silicon MPS
device = torch.device('mps')

# Set seeds for reproducibility
def set_seeds(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def select_datasets(files):
    # Define lower and test data
    lower_level = None
    target = None

    for file in files.iterdir():
        if 'Lower' in file.name:
            lower_level = file
        elif 'Target' in file.name:
            target = file

    return lower_level, target


def preprocess_csv(csv_file, return_origin_df_len=False):
    """
    Function to preprocess a csv file.
    Imports csv, refactors label column and masks NA values, aggregates into wide format
    applies sliding window and deletes unlabelled rows
    """
    preprocessed_df = pd.read_csv(csv_file)

    # Drop rows where 'Info_split' is missing or 'NA'
    if 'Info_split' in preprocessed_df.columns:
        preprocessed_df = preprocessed_df[
            preprocessed_df['Info_split'].notna() &
            (preprocessed_df['Info_split'] != 'NA')
            ]

    # Delete unlabelled rows
    if 'Class' in preprocessed_df.columns:
        preprocessed_df = preprocessed_df[
            preprocessed_df['Class'].notna() &
            (preprocessed_df['Class'] != 'NA')
            ]

    # Refactor class label
    preprocessed_df['Class'] = preprocessed_df['Class'].replace(-1, 0)

    return preprocessed_df


def pfeaturizer(df):
    """
    Function to add concatenated pfeatures to a dataframe
    """

    feature_functions = [
        ('aac', aac_wp),
        ('atc', atc_wp),
        ('btc', btc_wp),
        ('pcp', pcp_wp),
        ('sep', sep_wp),
        ('ctc', ctc_wp)
    ]

    all_features = []

    with tempfile.TemporaryDirectory() as tmpdir:

        for seq in tqdm(
                df['Info_window'],
                total=len(df),
                desc='Pfeaturizing input'
                ):

            seq_features = []

            fasta_file = os.path.join(tmpdir, 'sequence.fasta')

            with open(fasta_file, "w") as f:
                f.write(">seq\n")
                f.write(seq + "\n")

            for name, func in feature_functions:

                output_file = os.path.join(tmpdir, f'{name}.csv')

                func(fasta_file, output_file)

                feature_df = pd.read_csv(output_file)

                seq_features.extend(feature_df.iloc[0].tolist())

            all_features.append(seq_features)

    output_df = df.copy()

    output_df['pfeature_embeddings'] = all_features

    return output_df


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


def class_weighting_for_clf(lower_stacked):
    # Convert to csv
    df_lower_stacked = pd.read_csv(lower_stacked)

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


def train_one_epoch(model, embeddings, labels, optimiser, loss_function):
    model.train()
    optimiser.zero_grad()

    outputs = model(embeddings)

    loss = loss_function(outputs, labels)
    loss.backward()

    optimiser.step()

    preds = torch.argmax(outputs, dim=1)

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy_score(labels, preds)
    }

    return metrics


def val_one_epoch(model, embeddings, labels, loss_function):
    model.eval()

    with torch.no_grad():
        outputs = model(embeddings)
        loss = loss_function(outputs, labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

    val_preds = preds.cpu().numpy()
    val_labels = labels.cpu().numpy()
    val_probs = probs.cpu().numpy()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy_score(val_labels, val_preds),
        "precision": precision_score(val_labels, val_preds),
        "recall": recall_score(val_labels, val_preds),
        "f1": f1_score(val_labels, val_preds),
        "mcc": matthews_corrcoef(val_labels, val_preds),
        "auc": roc_auc_score(val_labels, val_probs)
    }

    return metrics, val_labels, val_probs, val_preds


def train_classifier(
        embedding_dim,
        train_dict,
        val_dict,
        loss_function,
        epochs=20):
    logs = []
    fold_labels_probs_preds_epoch = {}

    for fold in train_dict.keys():

        model = PerResidueClassifier(embedding_dim).to(device)

        optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            train_metrics = train_one_epoch(
                model,
                train_dict[fold]["embeddings"],
                train_dict[fold]["labels"],
                optimiser,
                loss_function)

            val_metrics, val_labels, val_probs, val_preds = val_one_epoch(model,
                                                                          val_dict[fold]["embeddings"],
                                                                          val_dict[fold]["labels"],
                                                                          loss_function)

            logs.append({'fold': fold,
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

            fold_labels_probs_preds_epoch[(fold, epoch + 1)] = (val_labels, val_probs, val_preds)

    history = pd.DataFrame(logs)

    return history, fold_labels_probs_preds_epoch


def train_final_classifier(model,
                           training_full,
                           loss_function,
                           epochs=20):
    logs = []

    model = model.to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_metrics = train_one_epoch(model,
                                        training_full['embeddings'],
                                        training_full['labels'],
                                        optimiser,
                                        loss_function)

        logs.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy']
        })

    history = pd.DataFrame(logs)

    return {
        'history': history,
        'model_state_dict': model.state_dict(),
        'epochs': epochs,
    }


