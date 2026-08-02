from train_pipeline import esm2_pipeline

CHECKPOINTS = [
    'facebook/esm2_t6_8M_UR50D',
    'facebook/esm2_t12_35M_UR50D',
    'facebook/esm2_t30_150M_UR50D',
    'facebook/esm2_t33_650M_UR50D',
    # 'facebook/esm2_t36_3B_UR50D',
]

MODES = [
    "base",
    "full",
    "LoRA"
]

files = "s3://esm2-s3-bucket/datasets/Orthopoxvirus/"

results = []

batch_size = 32

# Number of labels (positive and negative)
num_labels = 2

# Number of fine-tuning epochs
num_fine_tune_epochs = 3

# Number of classifier training epochs
clf_epochs = 20

for checkpoint in CHECKPOINTS:

    for mode in MODES:

        print("\n" + "="*80)
        print(f"Running {checkpoint} - {mode}")
        print("="*80)


        result = esm2_pipeline(
            checkpoint=checkpoint,
            mode=mode,
            num_labels=num_labels,
            files=files,
            fine_tune_val_folds=3,
            batch_size=batch_size,
            clf_epochs=clf_epochs,
            output_dir=
        )

        results.append(result)

print("All experiments completed")