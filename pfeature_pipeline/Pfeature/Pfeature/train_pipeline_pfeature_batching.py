# import Pfeature
import torch
import pandas as pd
import numpy as np
import torch.nn as nn


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix)

from pathlib import Path
from pathlib import PurePosixPath

import os
import tempfile
import gc

# Set device to Apple silicon MPS
device = torch.device('mps')

from utils_pfeature_copy import (select_datasets, preprocess_csv, class_weighting_for_clf,
    PerResidueClassifier, train_classifier, train_final_classifier, set_seeds, pfeaturizer)


def pipeline_pfeature(
                  clf_epochs,
                  pathogen,
                  batch_size):

    # Extract names
    pathogen_name = pathogen.rstrip('/').split('/')[-1]

    # Local output directory
    local_output = Path('./output_weighted_copy') / pathogen_name / 'Pfeature'

    local_output.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    set_seeds(42)

    files = Path(pathogen)

    # Select datasets and assign to variables for automatic csv selection in code
    lower_level, target = select_datasets(files)




    # ----------------- Preprocess lower level data for training classifier head  --------------------

    print('Running lower level data')

    # Preprocess the lower level data to create wide format dataframe with aggregated data by protein id
    df_lower = preprocess_csv(lower_level)

    # Extract pfeature features
    df_lower = pfeaturizer(df_lower)

    print('Lower Level Pfeaturizer complete', flush=True)

    lower_train_dict = {}
    lower_val_dict = {}

    lower_train_dict = {}
    lower_val_dict = {}

    for fold in range(1, 6):

        split = f'split_{fold:02d}_20'

        train_df = df_lower[df_lower['Info_split'] != split].reset_index(drop=True)
        val_df = df_lower[df_lower['Info_split'] == split].reset_index(drop=True)

        lower_train_dict[fold] = {
            'embeddings': torch.tensor(
                train_df['pfeature_embeddings'].tolist(),
                dtype=torch.float32,
                device=device
            ),
            'labels': torch.tensor(
                train_df['Class'].values,
                dtype=torch.long,
                device=device
            )
        }

        lower_val_dict[fold] = {
            'embeddings': torch.tensor(
                val_df['pfeature_embeddings'].tolist(),
                dtype=torch.float32,
                device=device
            ),
            'labels': torch.tensor(
                val_df['Class'].values,
                dtype=torch.long,
                device=device
            )
        }


    # Full training tensor
    lower_train_all = {
        'embeddings': torch.tensor(
            df_lower['pfeature_embeddings'].tolist(),
            dtype=torch.float32,
            device=device
        ),
        'labels': torch.tensor(
            df_lower['Class'].values,
            dtype=torch.long,
            device=device
        )
    }

    # Set a variable for the embedding dimension to use in the classifier
    embedding_dim = lower_train_all['embeddings'].shape[1]


    # Determine pos / neg weighting for loss function
    weight = class_weighting_for_clf(lower_level)
    weight = weight.to(device)


    # Instantiate classifier
    clf = PerResidueClassifier(embedding_dim)

    # Move to CUDA
    clf = clf.to(device)

    # Instantiate weighted cross-entropy loss function, ignores positions with mask -100
    loss_fcn = nn.CrossEntropyLoss( ignore_index=-100, weight=weight)

    train_rows = []
    val_rows = []

    for fold in lower_train_dict:

        # Training labels
        train_labels = lower_train_dict[fold]["labels"].detach().cpu().numpy()

        train_counts = pd.Series(train_labels).value_counts().sort_index()

        for label, count in train_counts.items():
            train_rows.append({
                "fold": fold,
                "label": label,
                "count": count
            })

        # Validation labels
        val_labels = lower_val_dict[fold]["labels"].detach().cpu().numpy()

        val_counts = pd.Series(val_labels).value_counts().sort_index()

        for label, count in val_counts.items():
            val_rows.append({
                "fold": fold,
                "label": label,
                "count": count
            })

    train_distribution = pd.DataFrame(train_rows)
    val_distribution = pd.DataFrame(val_rows)

    train_distribution.to_csv("train_label_distribution.csv", index=False)
    val_distribution.to_csv("val_label_distribution.csv", index=False)


    # -------- Train the classifier ----------------
    clf_trained, fold_labels_prob_preds = train_classifier(embedding_dim,
                                                           lower_train_dict,
                                                           lower_val_dict,
                                                           loss_fcn,
                                                           clf_epochs,
                                                           batch_size=batch_size)

    # Calculate average validation metrics per epoch
    clf_validation_avg_metrics = clf_trained.groupby(['epoch'])[[
        'train_loss',
        'train_acc',
        'val_loss',
        'val_acc',
        'val_precision',
        'val_recall',
        'val_f1',
        'val_mcc',
        'val_auc']].agg(['mean', 'std'])

    # Save to csv
    clf_validation_avg_metrics.to_csv(Path(f'{local_output}_clf_validation_avg_metrics.csv'))

    # Validation metrics per epoch by fold
    clf_validation_metrics = clf_trained[[
        'epoch',
        'fold',
        'train_loss',
        'train_acc',
        'val_loss',
        'val_acc',
        'val_precision',
        'val_recall',
        'val_f1',
        'val_mcc',
        'val_auc']]

    # Save to csv
    clf_validation_metrics.to_csv(Path(f'{local_output}_clf_validation_metrics_by_fold.csv'))

    best_auc_epochs = clf_validation_avg_metrics['val_auc'].idxmax(axis=0)['mean']

    # Save the fold, epoch val labels and val pred values for the best AUC
    rows = []
    for fold in lower_val_dict.keys():
        labels, probs, preds = fold_labels_prob_preds[(fold, best_auc_epochs)]
        rows.append(pd.DataFrame(
            {'Fold': fold,
             'Label': labels,
             'Prob': probs,
             'Preds': preds}))

    pd.concat(rows, ignore_index=True).to_csv(Path(f'{local_output}_cv_roc_epoch.csv'), index=False)

    print('clf_validation_avg_metrics', clf_validation_avg_metrics)

    clf = PerResidueClassifier(embedding_dim).to(device)



    # Train classifier using all data for the best AUC number of epochs
    final_clf_trained = train_final_classifier(clf,
                                               lower_train_all,
                                               loss_fcn,
                                               epochs=best_auc_epochs,
                                               batch_size=batch_size)

    torch.save(
    {
            'model_state_dict': final_clf_trained['model_state_dict'],
            'epochs': final_clf_trained['epochs'],
        },
        Path(f'{local_output}_final_classifier.pt'),
    )

    # Save results to csv
    final_clf_trained['history'].to_csv(Path(f'{local_output}_clf_final_trained_metrics_by_epoch.csv'))

    print('Final Trained Model Metrics', final_clf_trained)




    # ------------------------------------------ Predictions on test data ------------------------------------------

    print('Running target data')

    # -----------  Preprocess training data -----------

    test_df = preprocess_csv(target)

    # Generate pfeatures
    test_df = pfeaturizer(test_df)

    test_all = {
        'embeddings': torch.tensor(
            test_df['pfeature_embeddings'].tolist(),
            dtype=torch.float32,
            device=device
        ),
        'labels': torch.tensor(
            test_df['Class'].values,
            dtype=torch.long,
            device=device)}



    # ------ Evaluate using pretrained classifier --------

    # Instantiate new model
    clf = PerResidueClassifier(embedding_dim).to(device)

    # Load trained clf model
    checkpoint_data = torch.load(Path(f'{local_output}_final_classifier.pt'), weights_only=False)

    # Load trained weights into clf
    clf.load_state_dict(checkpoint_data['model_state_dict'])

    # Put classifier into evaluation mode
    clf.eval()

    test_preds = []
    test_labels = []
    test_probs = []

    with torch.no_grad():

        outputs = clf(test_all['embeddings'].to(device))

        labels = test_all['labels'].reshape(-1).to(device)

        # Calculate preds and probs
        preds = torch.argmax(outputs, dim=-1)
        probs = torch.softmax(outputs, dim=-1)[:, 1]

        test_preds.append(preds)
        test_labels.append(labels)
        test_probs.append(probs)

    # Concat tensors
    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)
    test_probs = torch.cat(test_probs)

    test_preds = test_preds.cpu().numpy()
    test_labels = test_labels.cpu().numpy()
    test_probs = test_probs.cpu().numpy()

    # Save test labels and probs
    test_labels_probs_preds = pd.DataFrame({
        'protein_id': test_df['Info_protein_id'].values,
        'position': test_df['Info_pos'].values,
        'label': test_labels,
        'probs': test_probs,
        'preds': test_preds,
    })

    test_labels_probs_preds.to_csv(Path(f'{local_output}_test_roc.csv'), index=False)

    # Calculate metrics
    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds)
    recall = recall_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)
    mcc = matthews_corrcoef(test_labels, test_preds)
    auc = roc_auc_score(test_labels, test_probs)

    conf_matrix = confusion_matrix(test_labels, test_preds)

    conf_df = pd.DataFrame(
        conf_matrix,
        index=['Actual 0', 'Actual 1'],
        columns=['Predicted 0', 'Predicted 1'],
    )

    # Save to CSV
    conf_df.to_csv(Path(f'{local_output}_test_confusion_matrix.csv'))

    # Create pandas dataframe
    data = { 'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, 'F1':f1, 'mcc':mcc, 'auc':auc}

    test_results = pd.DataFrame(data, index=np.array(np.arange(1,2)))

    # Save to csv
    test_results.to_csv(Path(f'{local_output}_test_predictions.csv'))

    print('test results', test_results, flush=True)

    # Free up memory
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {'pathogen': pathogen}

