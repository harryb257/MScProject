from train_pipeline_pfeature_copy import pipeline_pfeature

pathogens = [
'/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Bunyaviricetes',
# '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Campylobacter jejuni',
# '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Clostridioides difficile',
# '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Dengue virus',
# '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Legionellales',
# '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Mycobacterium leprae',
# '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Mycoplasmoidaceae',
# '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Neisseria gonorrhoeae',
# '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Orthoparamyxovirinae',
# '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Orthopoxvirus',
# '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Pasteurellaceae',
# #'/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Yellow fever virus',
# '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/pfeature_pipeline/Pfeature/datasets/Yersinia pestis'
]

results = []

# Number of classifier training epochs
clf_epochs = 20

# Dataset batch size for training classifier
batch_size = 16

for pathogen in pathogens:
    print(f"Running {pathogen}")

    result = pipeline_pfeature(
        clf_epochs=clf_epochs,
        pathogen=pathogen,
        batch_size=batch_size,
    )

    results.append(result)
    print('Runs completed:', results)

print("All experiments completed")
