import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

import datasets

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
    roc_auc_score,
    confusion_matrix)

from pathlib import Path
import s3fs
import boto3
import gc

from utils_prott5 import (load_prott5_with_classification_head, select_datasets, preprocess_csv,
    preprocess_higher_level, trainable_parameters_summary, compute_metrics, SequenceDataset, batch_create,
    class_weighting_for_clf, PerResidueClassifier, create_datasets_for_clf, train_classifier, train_final_classifier,
    delete_unlabelled_rows, sliding_window, set_seeds)

from peft import PeftModel, PeftConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prott5_pipeline(checkpoint,
                  mode,
                  num_labels,
                  files,
                  fine_tune_val_folds,
                  batch_size,
                  num_fine_tune_epochs,
                  clf_epochs,
                  pathogen):

    # Extract names
    pathogen_name = pathogen.rstrip('/').split('/')[-1]
    checkpoint_name = checkpoint.split('/')[-1]

    # Local output directory
    local_output = Path(f'./output/{pathogen_name}/{checkpoint_name}_{mode}')

    local_output.mkdir(parents=True, exist_ok=True)

    # S3 destination
    bucket = 'esm2-s3-bucket'
    prefix = f'results_prott5'
    s3 = boto3.client('s3')

    # Set random seeds
    set_seeds(42)

    # # Set fine-tuning method
    # if mode == 'full':
    #     full_fine_tuning = True
    # elif mode == 'LoRA':
    #     full_fine_tuning = False

    # Set global embedding dimension variable, used for PerResidueClassifier input
    embedding_dim = 1024

    # Model selection
    if mode == 'base':
        tokenizer = T5Tokenizer.from_pretrained(checkpoint, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(checkpoint)
    elif mode == 'LoRA':
        tokenizer, model = load_prott5_with_classification_head(checkpoint, mode, embedding_dim)
    elif mode == 'full':
        model, tokenizer = load_prott5_with_classification_head(checkpoint, mode, embedding_dim)

    # Set model to device
    model.to(device)

    print(model)

    # Select S3 datasets
    fs = s3fs.S3FileSystem()

    files = fs.ls(files)

    # Assign to variables for automatic csv selection in code
    lower_level, higher_level, target = select_datasets(files)




    # -------------- Fine tuning Preprocessing -----------------
    if mode != 'base':

        # Load and preprocess dataframe
        df_higher = preprocess_csv(higher_level)

        train_splits = {}
        val_splits = {}

        for fold in range(1, fine_tune_val_folds+1):
            split = f'split_{fold:02d}_20'

            train_splits[fold] = (df_higher[df_higher['Info_split'] != split].reset_index(drop=True))

            val_splits[fold] = (df_higher[df_higher['Info_split'] == split].reset_index(drop=True))

        # Create split for final fine-tuning using all data
        train_splits['final'] = df_higher.reset_index(drop=True)

        # No validation set for the final model
        val_splits['final'] = None

        train_datasets = {}
        val_datasets = {}

        for key, df in train_splits.items():
            train_datasets[key] = (datasets.Dataset.from_pandas(df).map(
                     		        lambda data: preprocess_higher_level(tokenizer, data),
					                remove_columns=df.columns.tolist()))

        for key, df in val_splits.items():
            if df is not None:
                val_datasets[key] = (datasets.Dataset.from_pandas(df).map(
					                lambda data: preprocess_higher_level(tokenizer, data),
                                    remove_columns=df.columns.tolist()))
            else:
                val_datasets[key] = None

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Generate trainable parameters summary for given mode
    parameter_summary = trainable_parameters_summary(model)

    # Save parameter summary to csv
    parameter_summary.to_csv(f'{local_output}_trainable_parameters_summary.csv')




    # -------------- Fine tuning -----------------

    # Store results
    train_logs = []
    eval_logs = []
    summary_logs = []

    final_train_logs = []
    final_summary_logs = []

    if mode != 'base':

        for fold in range(1, fine_tune_val_folds+1):

            # Load fresh model for each train/validation fold
            model, tokenizer = load_esm_model_classification(checkpoint, num_labels, full_fine_tuning)

            training_args = TrainingArguments(
                output_dir=Path(f'{local_output}_training_arguments_val'),
                gradient_accumulation_steps=1,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                fp16=True,
                num_train_epochs=num_fine_tune_epochs,
                learning_rate=3e-4, # Match to Nature peft paper
                seed=42,
                save_strategy='no',
                eval_strategy='epoch',
                logging_strategy='epoch',
                load_best_model_at_end=False,
                metric_for_best_model='auc',
                greater_is_better=True,
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
                dataloader_persistent_workers=True
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_datasets[fold],
                eval_dataset=val_datasets[fold],
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )

            trainer.train()

            for log in trainer.state.log_history:
                log = {**log, 'fold': fold}

                if 'eval_loss' in log:
                    eval_logs.append(log)
                elif 'grad_norm' in log:
                    train_logs.append(log)
                elif 'train_runtime' in log:
                    summary_logs.append(log)

        eval_log_df = pd.DataFrame(eval_logs)
        train_log_df = pd.DataFrame(train_logs)
        summary_df = pd.DataFrame(summary_logs)

        # Train on full data -----------------------

        # Load fresh model
        model, tokenizer = load_esm_model_classification(checkpoint, num_labels, full_fine_tuning)

        training_args = TrainingArguments(
            output_dir = Path(f'{local_output}_finetune_training_final'),
            gradient_accumulation_steps=1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_fine_tune_epochs,
            learning_rate=3e-4, # Match to Nature peft paper
            seed=42,
            fp16=True,
            eval_strategy='no',
            save_strategy='no',
            logging_strategy='epoch',
            load_best_model_at_end=False,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_datasets['final'],
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        for log in trainer.state.log_history:

            if "loss" in log:
                final_train_logs.append(log)
            elif "train_runtime" in log:
                final_summary_logs.append(log)

        # Convert to DataFrames
        final_train_log_df = pd.DataFrame(final_train_logs)
        final_summary_df = pd.DataFrame(final_summary_logs)

        model_output = Path(f'{local_output}_fine_tuned_model')

        # Explicitly save model
        trainer.save_model(model_output)

        # Save validation metrics
        eval_log_df.to_csv(Path(f'{local_output}_cv_validation_metrics.csv'), index=False)
        train_log_df.to_csv(Path(f'{local_output}_cv_train_metrics.csv'), index=False)
        summary_df.to_csv(Path(f'{local_output}_cv_eval_log.csv'), index=False)

        # Save final logs
        final_train_log_df.to_csv(Path(f'{local_output}_final_training_log.csv'), index=False)
        final_summary_df.to_csv(Path(f'{local_output}_final_training_summary.csv'), index=False)

    # -------------------------------------




    # ----------------- Preprocess lower level data for training classifier head  --------------------

    # Generate splits (splits will also be used to train the classification head for fine-tuning pipelines)
    df_lower = preprocess_csv(lower_level)

    # Create train / validation splits using the info split column
    lower_train_dict= {}
    lower_val_dict = {}
    lower_train_all = {}

    for fold in range(1, 6):
        split = f'split_{fold:02d}_20'

        lower_train_dict[fold] = (df_lower[df_lower['Info_split'] != split].reset_index(drop=True))

        lower_val_dict[fold] = (df_lower[df_lower['Info_split'] == split].reset_index(drop=True))

    # Create split of data for final training
    lower_train_all = df_lower.reset_index(drop=True)
    # Create dataset
    lower_train_all = SequenceDataset(lower_train_all)


    if mode != 'base':
        if mode == 'full':
            model = AutoModelForTokenClassification.from_pretrained(Path(f'{local_output}_fine_tuned_model'))

        elif mode == 'LoRA':
            base_model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=num_labels)
            model = PeftModel.from_pretrained(base_model, Path(f'{local_output}_fine_tuned_model'))

        model.eval()
        model.to(device)

        # Generate embeddings for lower level data using the fine-tuned model
        train_loaded, val_loaded = create_datasets_for_clf(lower_train_dict, lower_val_dict, batch_size,
                                                           tokenizer, model, mode)

        full_train_batched = batch_create(lower_train_all, batch_size, tokenizer, model, mode)

    else:
        # Generate embeddings for lower level data using the base model
        train_loaded, val_loaded = create_datasets_for_clf(lower_train_dict, lower_val_dict, batch_size,
                                                           tokenizer, model, mode)

        full_train_batched = batch_create(lower_train_all, batch_size, tokenizer, model, mode)



    # Determine pos / neg weighting for loss function
    weight = class_weighting_for_clf(lower_level)
    weight = weight.to(device)


    # Create a parameter for the final hidden layer dim of the model to use as the classifier input dimension
    embedding_dim = model.config.hidden_size

    # Instantiate classifier
    clf = PerResidueClassifier(embedding_dim)

    # Move to CUDA
    clf = clf.to(device)

    # Instantiate weighted cross-entropy loss function, ignores positions with mask -100
    loss_fcn = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)




    # -------- Train the classifier ----------------
    clf_trained, fold_labels_prob_preds = train_classifier(embedding_dim,
                                                     train_loaded,
                                                     val_loaded,
                                                     loss_fcn,
                                                     epochs=clf_epochs)

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

    # Calculate the epoch with the highest AUC value averaged across the folds
    best_auc_epochs = clf_validation_avg_metrics['val_auc'].idxmax(axis=0)['mean']

    # Save the fold, epoch val labels and val pred values for the best AUC
    rows = []
    for fold in val_loaded.keys():
        labels, probs, preds = fold_labels_prob_preds[(fold, best_auc_epochs)]
        rows.append(pd.DataFrame(
            {'Fold': fold,
             'Label': labels,
             'Prob': probs,
             'Preds': preds}))
    pd.concat(rows, ignore_index=True).to_csv(Path(f'{local_output}_cv_roc_epoch.csv'), index=False)

    print('clf_validation_avg_metrics', clf_validation_avg_metrics)

    # Instantiate new classifier
    clf = PerResidueClassifier(embedding_dim).to(device)

    # Train classifier using all data for the best AUC number of epochs
    final_clf_trained = train_final_classifier(clf, full_train_batched, loss_fcn, epochs=best_auc_epochs)

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

    # Preprocess training data
    test_df = pd.read_csv(target)

    # Mask n/a values with -100
    test_df['Class'] = test_df['Class'].fillna(-100).astype('int32')
    test_df['Class'] = test_df['Class'].replace(-1, 0)

    # Sort values before aggregation
    test_df = test_df.sort_values(['Info_protein_id', 'Info_pos'])

    # Aggregate columns for wide format
    test_df = test_df.groupby(['Info_protein_id'], as_index=False).agg(
        sequence=('Info_AA', ''.join),
        label=('Class', list),
        position=('Info_pos', list))

    # Apply sliding window
    test_df = sliding_window(test_df)

    # Delete rows
    test_df = delete_unlabelled_rows(test_df)

    # Create datasets
    test_datasets = SequenceDataset(test_df)

    # Create batches and embeddings using relevant ESM2 model
    test_batched = batch_create(test_datasets, batch_size, tokenizer, model, mode)


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

        for batch in test_batched:
            outputs = clf(batch['embeddings'].to(device))

            labels = batch['labels'].reshape(-1).to(device)

            # Calculate preds and probs
            preds = torch.argmax(outputs, dim=-1).reshape(-1)
            probs = torch.softmax(outputs, dim=-1)[:, :, 1].reshape(-1)

            test_preds.append(preds)
            test_labels.append(labels)
            test_probs.append(probs)

    # Concat tensors
    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)
    test_probs = torch.cat(test_probs)

    mask = test_labels != -100

    test_preds = test_preds[mask].cpu().numpy()
    test_labels = test_labels[mask].cpu().numpy()
    test_probs = test_probs[mask].cpu().numpy()

    # Save test labels and probs
    test_labels_probs_preds = pd.DataFrame({
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

    print('test=', confusion_matrix(test_labels, test_preds))

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

    print('test results', test_results)

    local_output2 = Path("./output")

    # Upload everything under local_output
    for file in local_output2.rglob("*"):
        if file.is_file():
            key = f'{prefix}/{file.relative_to(local_output2).as_posix()}'
            print('uploading', file)
            print('S3 key', key)
            try:
                s3.upload_file(str(file), bucket, key)
                print('Upload complete:', key)

                # Delete local file after successful upload
                file.unlink()
                print('Deleted local file:', file)

            except Exception as e:
                print('Upload failed:', e)

    # Remove empty directories
    for d in sorted(local_output2.rglob('*'), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                # Directory not empty
                pass


    # Free up memory
    gc.collect()
    torch.cuda.empty_cache()


    return {
        'pathogen': pathogen,
        'checkpoint': checkpoint,
        'mode': mode
    }



