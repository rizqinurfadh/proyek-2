import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import zscore

# Membaca dataset baru
df = pd.read_csv('modified_dataset.csv')
print(df)

# Mengganti nama kolom
df = df.rename(columns={'PTN': 'ptn', 'NAMA PRODI': 'program_studi', 'MIN': 'passing_grade', 'SKOR': 'skor', 'KELAYAKAN': 'kelayakan'})
print(df)

# MISSING VALUE
# Mengecek jumlah nilai yang hilang untuk setiap kolom
missing_values = df.isnull().sum()
print("Jumlah missing values untuk setiap kolom:")
print(missing_values) 

# OUTLIER
# Menghitung z-score untuk setiap nilai dalam dataset
z_scores = zscore(df[['passing_grade', 'skor']])

# Mengambil nilai z-score absolut untuk mengidentifikasi outliers
z_scores_abs = np.abs(z_scores)

# Menentukan threshold untuk outlier
threshold = 3

# Mengidentifikasi baris yang memiliki nilai z-score > threshold
outliers_mask = (z_scores_abs > threshold).any(axis=1)

# Menampilkan baris yang mengandung outliers
print("Baris yang mengandung outliers:")
print(df[outliers_mask])

# Menghapus baris yang mengandung outliers dari dataframe
df = df[~outliers_mask]

# Menampilkan dataframe setelah penghapusan outlier
print("\nDataframe setelah penghapusan outlier:")
print(df)

# Mengubah data kategori menjadi data numerik
label_encoder_ptn = LabelEncoder()
df['ptn'] = label_encoder_ptn.fit_transform(df['ptn'])

label_encoder_prodi = LabelEncoder()
df['program_studi'] = label_encoder_prodi.fit_transform(df['program_studi'])

label_encoder_status = LabelEncoder()
df['kelayakan'] = label_encoder_status.fit_transform(df['kelayakan'])

# Memisahkan fitur dan target
x = df[['ptn', 'program_studi', 'passing_grade', 'skor']]
y = df['kelayakan']

# Membagi dataset menjadi data pelatihan dan data pengujian
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(df)
print(df.dtypes)

# PEMODELAN Random Forest
# Inisialisasi model Random Forest
model_rf = RandomForestClassifier(random_state=21, n_estimators=100, max_depth=5)

# Melatih model dengan data pelatihan
model_rf.fit(x_train, y_train)

# Cross-validation untuk mengevaluasi kinerja model
scores = cross_val_score(model_rf, x_train, y_train, cv=5, scoring='accuracy')
print("Akurasi model Random Forest (cross-validation):", np.mean(scores) * 100, "%")

# Minta pengguna untuk memasukkan input
ptn =   int(input("Masukkan nama PTN: "))
program_studi = int(input("Masukkan nama program studi: "))
skor = int(input("Masukkan skor: "))

# Cari passing grade berdasarkan PTN dan program studi
filtered_df = df[(df['ptn'] == ptn) & (df['program_studi'] == program_studi)]
if filtered_df.empty:
    print("Maaf, data tidak tersedia untuk PTN dan program studi yang dimasukkan.")
else:
    passing_grade = filtered_df['passing_grade'].values[0]

# Buat array fitur baru sesuai dengan nama fitur yang digunakan saat melatih model
skor_new = [[passing_grade, program_studi, ptn, skor]]

# Prediksi dengan model Random Forest
predicted_label_code = model_rf.predict(skor_new) 

# Invers transformasi nilai kelas prediksi menjadi kategori semula
predicted_label_name = label_encoder_status.inverse_transform(predicted_label_code)

print("Prediksi kelas label untuk skor yang dimasukkan pengguna:")
print(predicted_label_name)