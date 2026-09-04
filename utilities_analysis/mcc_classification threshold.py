import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef


pathogen = 'LoRA'
modes = ['base','LoRA', 'full']
sizes = ['8M','35M','150M','650M','1.2B']

mode = modes[2]
size = sizes[3]

print(mode, size)

# roc_data_clf_eval = pd.read_csv(f'/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/Results_ESM2_weighted_clf/{pathogen}/esm2_t33_{size}_UR50D_{mode}_cv_roc_epoch.csv')
roc_data_clf_eval = pd.read_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/Results_ProtT5/Orthopoxvirus/prot_t5_xl_half_uniref50-enc_LoRA_cv_roc_epoch.csv')
# roc_data_clf_eval = pd.read_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/Pfeature/output_weighted_batched/Dengue virus/Pfeature_cv_roc_epoch.csv')

y_prob = roc_data_clf_eval['Prob']
y_true = roc_data_clf_eval['Label']

# Span of thresholds to test
thresholds = np.linspace(0, 1, 101)
mcc_scores = []

for threshold in thresholds:
    # Convert probabilities into boolean values using the applied threshold value
    y_pred = (y_prob >= threshold).astype(int)
    # Calculate MCC for this threshold with the new y_preds
    mcc = matthews_corrcoef(y_true, y_pred)
    mcc_scores.append(mcc)

# Get highest mcc index
best_threshold_num = np.argmax(mcc_scores)
# Return threshold value for this index
best_threshold = thresholds[best_threshold_num]
# Return mcc for the best threshold
best_mcc = mcc_scores[best_threshold_num]


# Update the target prediction MCC scores with the best threshold from the clf
# roc_data_test = pd.read_csv(f'/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/Results_ESM2_weighted_clf/{pathogen}/esm2_t33_{size}_UR50D_{mode}_test_roc.csv')

roc_data_test = pd.read_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/Results_ProtT5/Orthopoxvirus/prot_t5_xl_half_uniref50-enc_LoRA_test_roc.csv')
# roc_data_test = pd.read_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/Pfeature/output_weighted_batched/Dengue virus/Pfeature_test_roc.csv')

y_prob_test = roc_data_test['probs']
y_true_test = roc_data_test['label']

mcc_scores_test = []
for threshold in thresholds:
    # Convert probabilities into boolean values using the applied threshold value
    y_pred_test = (y_prob_test >= threshold).astype(int)
    # Calculate MCC for this threshold with the new y_preds
    mcc_test = matthews_corrcoef(y_true_test, y_pred_test)
    mcc_scores_test.append(mcc_test)

# Get highest mcc index
best_threshold_num_test = np.argmax(mcc_scores_test)
# Return threshold value for this index
best_threshold_test = thresholds[best_threshold_num_test]
# Return mcc for the best threshold
best_mcc_test = mcc_scores_test[best_threshold_num_test]

fig, ax = plt.subplots(1, 1)
ax.plot(thresholds, mcc_scores, label='Classifer lower-level')
ax.plot(thresholds, mcc_scores_test, label='Target level')
ax.set_title(f'{pathogen} \n Classifier best threshold: {best_threshold:.2f}, Best MCC: {best_mcc:.2f}'
             f'\n Target best threshold: {best_threshold_test:.2f}, Best MCC: {best_mcc_test:.2f}')
ax.legend()
plt.xlabel('Threshold')
plt.ylabel('MCC')
# Best threshold line
plt.axvline(best_threshold, linestyle=':')
plt.axvline(best_threshold_test, linestyle=':', color='orange')
plt.show()


