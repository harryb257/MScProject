import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from sklearn.metrics import roc_curve, roc_auc_score

merged_results_folder = Path('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/Pfeature/output_weighted_batched/merged_results')

roc_csv_file = merged_results_folder / 'cv_roc_epoch.csv'

output_dir = merged_results_folder / 'roc_auc_plots_clf_cv'
output_dir.mkdir(exist_ok=True)

test_roc_data = pd.read_csv(roc_csv_file)

pathogens = test_roc_data['pathogen'].unique()


def plot_roc(dataset, pathogen):
    """
    Function to generate roc curve with auc score for a given pathogen and mode
    """

    subset = dataset[dataset['pathogen'] == pathogen]

    fig, ax = plt.subplots(figsize=(6, 6))

    for fold in sorted(subset['Fold'].unique()):
        model_subset = subset[subset['Fold'] == fold]

        y_true = model_subset['Label']
        y_score = model_subset['Prob']

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)

        ax.plot(fpr, tpr, label=f'Fold: {fold} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Classifier Cross Validation (Pfeature) \n ROC Curves \n {pathogen}')
    ax.legend(title='Model Size')
    ax.grid(alpha=0.3)

    fig.tight_layout()

    # return fig


for pathogen in pathogens:
    fig = plot_roc(test_roc_data, pathogen)

    output = Path(output_dir) / f'{pathogen}.png'

    plt.savefig(
        output,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(fig)

    print('Saved plot:', pathogen)
