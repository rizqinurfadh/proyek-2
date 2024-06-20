import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import zscore

# Membaca dataset
data = pd.read_csv('passing-grade.csv')
print(data)             

# Menghapus kolom yang tidak dipakai
df = data.drop(columns=['NO', 'KODE PRODI', 'RATAAN', 'S.BAKU', 'MAX',])
print(df)

# Menambahkan kolom skor dengan nilai acak antara 500 sampai 810 
df['SKOR'] = np.random.randint(500, 810, size=len(df))

# Menambahkan kolom status berdasarkan nilai skor dan min
df['KELAYAKAN'] = np.where(df['SKOR'] >= df['MIN'], 'Cocok', 'Tidak Cocok')

# Menyimpan kembali dataset ke file CSV
output_path = 'modified_dataset.csv'
df.to_csv(output_path, index=False)
print(df)