import pandas as pd
from sklearn.model_selection import train_test_split

# Legge il file CSV originale
file_path = "data/image/rsna/train.csv"
df = pd.read_csv(file_path)
print(df.head())

# Estrai tutti gli StudyInstanceUID unici
unique_studies = df['StudyInstanceUID'].unique()

# Esegui lo split 80% train, 20% valid sugli StudyInstanceUID
train_studies, valid_studies = train_test_split(unique_studies, test_size=0.2, random_state=42)

# Assegna 'train' o 'valid' in base allo StudyInstanceUID
df['Split'] = df['StudyInstanceUID'].apply(lambda x: 'train' if x in train_studies else 'valid')

# Salva il file con la colonna Split
df.to_csv("data/image/rsna/rsna_train_master.csv", index=False)

print("rsna_train_master.csv generated")