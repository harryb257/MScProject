from train_pipeline import esm2_pipeline

CHECKPOINTS = [
    'facebook/esm2_t6_8M_UR50D',
    'facebook/esm2_t12_35M_UR50D',
    'facebook/esm2_t30_150M_UR50D',
    # 'facebook/esm2_t33_650M_UR50D',
    # 'facebook/esm2_t36_3B_UR50D',
]

MODES = [
    # "base",
    # "full",
    "LoRA"
]

pathogens = [
    # 's3://esm2-s3-bucket/datasets/Bunyaviricetes/',
    # 's3://esm2-s3-bucket/datasets/Campylobacter jejuni/',
    # 's3://esm2-s3-bucket/datasets/Clostridioides difficile/',
    # 's3://esm2-s3-bucket/datasets/Dengue virus/',
    # 's3://esm2-s3-bucket/datasets/Legionellales/',
    # 's3://esm2-s3-bucket/datasets/Mycobacterium leprae/',
    # 's3://esm2-s3-bucket/datasets/Mycoplasmoidaceae/',
    # 's3://esm2-s3-bucket/datasets/Neisseria gonorrhoeae/',
    # 's3://esm2-s3-bucket/datasets/Orthoparamyxovirinae/',
    # 's3://esm2-s3-bucket/datasets/Orthopoxvirus/',
    # 's3://esm2-s3-bucket/datasets/Pasteurellaceae/',
    's3://esm2-s3-bucket/datasets/Yellow fever virus/',
    's3://esm2-s3-bucket/datasets/Yersinia pestis/']

results = []

# Batch size to use throughout
batch_size = 16
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

            result = esm2_pipeline(
                checkpoint=checkpoint,
                mode=mode,
                num_labels=num_labels,
                files=pathogen,
                fine_tune_val_folds=3,
                batch_size=batch_size,
                num_fine_tune_epochs=num_fine_tune_epochs,
                clf_epochs=clf_epochs,
                pathogen=pathogen
            )

            results.append(result)
            print('Runs completed:', results)

print("All experiments completed")
