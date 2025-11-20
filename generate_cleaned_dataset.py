from cleaning import preprocess_dataframe

import zipfile

import pandas as pd

filepath = input("Input the path to the zipped dataset downloaded from Kaggle")

# to extract the zipfile downloaded from kaggle
with zipfile.ZipFile(filepath) as zip:
    zip.extract('train.csv.zip', path='./data')

# load the zipfile into a dataframe
def load_csv_from_zip(zip_path): 
    with zipfile.ZipFile(zip_path, 'r') as z: 
        csv_filename = z.namelist()[0] 
        with z.open(csv_filename) as f: 
            return pd.read_csv(f)
        
df_train = load_csv_from_zip('./data/train.csv.zip')

# to run the preprocessing pipeline
df_cleaned = preprocess_dataframe(df_train)

# lastly, to save the cleaned dataset
df_cleaned.to_csv('data/clean_toxic_comment_dataset.csv', index=False)
print("Cleaned Dataset Saved!")