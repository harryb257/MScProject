from train_pipeline_prott5 import prott5_pipeline
from pathlib import Path

CHECKPOINTS = [
    "Rostlab/prot_t5_xl_half_uniref50-enc"
]

MODES = [
    #"base",
    #"LoRA",
    "full",
    #"base"
]

pathogens = [
   # Path('./HB/datasets/Bunyaviricetes/'),
   # Path('./HB/datasets/Campylobacter jejuni/'),
   # Path('./HB/datasets/Clostridioides difficile/'),
   # Path('./HB/datasets/Dengue virus/'),
   # Path('./HB/datasets/Legionellales/'),
   # Path('./HB/datasets/Mycobacterium leprae/'),
   # Path('./HB/datasets/Mycoplasmoidaceae/'),
   # Path('./HB/datasets/Neisseria gonorrhoeae/'),
   # Path('./HB/datasets/Orthoparamyxovirinae/'),
    Path('./HB/datasets/Orthopoxvirus/'),
   # Path('./HB/datasets/Pasteurellaceae/'),
    # Path('./HB/datasets/Yellow fever virus/'),
   # Path('./HB/datasets/Yersinia pestis/')
]

results = []

# Batch size to use throughout
batch_size = 4
# Number of labels (positive and negative)
num_labels = 2
# Number of fine-tuning epochs
num_fine_tune_epochs = 3
# Number of classifier training epochs
clf_epochs = 20

for pathogen in pathogens:
    for checkpoint in CHECKPOINTS:
        for mode in MODES:
            print(f"Running {pathogen} - {checkpoint} - {mode}")

            result = prott5_pipeline(
                checkpoint=checkpoint,
                mode=mode,
                num_labels=num_labels,
                files=pathogen,
                fine_tune_val_folds=3,
                batch_size=batch_size,
                num_fine_tune_epochs=num_fine_tune_epochs,
                clf_epochs=clf_epochs,
            )

            results.append(result)
            print('Runs completed:', results)

print("All experiments completed")
