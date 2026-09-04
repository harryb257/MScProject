import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from sklearn.metrics import roc_curve, roc_auc_score

merged_results_folder = Path('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/Pfeature/output_weighted_batched/merged_results')

roc_csv_file = merged_results_folder / 'test_roc.csv'

output_dir = merged_results_folder / 'test_roc_auc_plot'
output_dir.mkdir(exist_ok=True)

test_roc_data = pd.read_csv(roc_csv_file)

pathogens = test_roc_data['pathogen'].unique()

# Line styles
line_styles = ['-', '--', '-.', ':']

fig, ax = plt.subplots(figsize=(10, 10))

for i, pathogen in enumerate(sorted(pathogens)):
    pathogen_subset = test_roc_data[test_roc_data['pathogen'] == pathogen]

    y_true = pathogen_subset['label']
    y_score = pathogen_subset['probs']

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    linestyle = line_styles[i % len(line_styles)]

    ax.plot(fpr, tpr, label=f'{pathogen} (AUC = {roc_auc:.3f})', linestyle=linestyle, linewidth=2)

ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'Target data (Pfeature) \n ROC Curves \n {pathogen}')
ax.legend(title='Model Size')
ax.grid(alpha=0.3)

fig.tight_layout()

plt.show()

output = Path(output_dir) / f'pathogens.png'

fig.savefig(
    output,
    dpi=300,
    bbox_inches='tight'
)