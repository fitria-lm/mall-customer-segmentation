# 🛍️ Mall Customer Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Analisis segmentasi pelanggan mall menggunakan algoritma Machine Learning untuk mengelompokkan pelanggan berdasarkan karakteristik demografi dan perilaku belanja.

## 📊 Demo & Visualisasi

| Elbow Method | 3D Clustering | Income vs Spending |
|:---:|:---:|:---:|
| ![Elbow Method](reports/figures/elbow_method.png) | ![3D Clustering](reports/figures/cluster_3d.png) | ![Income vs Spending](reports/figures/eda_plots.png) |

## 📋 Daftar Isi
- [Latar Belakang](#-latar-belakang)
- [Fitur Dataset](#-fitur-dataset)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Hasil Analisis](#-hasil-analisis)
- [Metodologi](#-metodologi)
- [Teknologi](#-teknologi)
- [Kontribusi](#-kontribusi)
- [Lisensi](#-lisensi)
- [Kontak](#-kontak)

## 🎯 Latar Belakang

Proyek ini bertujuan untuk melakukan segmentasi pelanggan mall berdasarkan data demografi (usia, jenis kelamin, pendapatan) dan perilaku belanja (spending score). Dengan segmentasi ini, tim pemasaran dapat:
- 🎯 Menargetkan kampanye iklan yang lebih personal
- 📈 Meningkatkan efektivitas strategi pemasaran
- 💡 Memahami pola perilaku pelanggan yang berbeda
- 🏷️ Mengembangkan program loyalitas yang sesuai

**Dataset**: [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) dari Kaggle (200 records)

## 📈 Fitur Dataset

| Kolom | Deskripsi | Tipe Data |
|-------|-----------|-----------|
| CustomerID | ID unik pelanggan | Integer |
| Genre | Jenis kelamin (Male/Female) | Categorical |
| Age | Usia pelanggan | Integer |
| Annual Income (k$) | Pendapatan tahunan (dalam ribuan USD) | Integer |
| Spending Score (1-100) | Skor belanja yang diberikan mall (1-100) | Integer |

**Statistik Deskriptif:**
- Jumlah pelanggan: 200
- Rata-rata usia: 38.85 tahun
- Rata-rata pendapatan: $60.56k
- Rata-rata spending score: 50.2

## 📁 Struktur Proyek

```
mall-customer-segmentation/
│
├── 📁 data/                           # Data storage
│   ├── 📁 raw/                       # Data mentah
│   │   └── Mall_Customers.csv        # Dataset asli
│   └── 📁 processed/                 # Data hasil proses
│       └── mall_customers_clustered.csv
│
├── 📁 notebooks/                      # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb     # Eksplorasi data
│   ├── 02_feature_analysis.ipynb     # Analisis fitur
│   └── 03_clustering.ipynb           # Modeling clustering
│
├── 📁 src/                           # Source code Python
│   ├── __init__.py
│   ├── data_loader.py               # Load & cek data
│   ├── preprocessing.py             # Cleaning & preprocessing
│   ├── eda.py                       # Exploratory Data Analysis
│   ├── clustering.py                # K-Means algorithm
│   ├── visualization.py             # Plotting functions
│   └── utils.py                     # Utility functions
│
├── 📁 reports/                       # Laporan & hasil
│   ├── 📁 figures/                  # Visualisasi
│   │   ├── eda_plots.png
│   │   ├── elbow_method.png
│   │   ├── cluster_3d.png
│   │   └── cluster_profiles.png
│   └── summary_report.md            # Laporan analisis
│
├── 📁 tests/                         # Unit tests
│   ├── test_preprocessing.py
│   └── test_clustering.py
│
├── 📄 requirements.txt               # Dependencies
├── 📄 requirements-dev.txt          # Dev dependencies
├── 📄 main.py                       # Script utama
├── 📄 config.py                     # Konfigurasi
└── 📄 README.md                     # Dokumentasi ini
```

## 🛠️ Instalasi

### Prasyarat
- Python 3.8 atau lebih tinggi
- pip atau conda

### 1. Clone Repository
```bash
git clone https://github.com/fitria-lm/mall-customer-segmentation.git
cd mall-customer-segmentation
```

### 2. Setup Virtual Environment (Rekomendasi)
```bash
# Menggunakan venv
python -m venv venv

# Aktifkan di Windows
venv\Scripts\activate

# Aktifkan di Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install semua dependencies utama
pip install -r requirements.txt

# Untuk development (opsional)
pip install -r requirements-dev.txt
```

### 4. Jalankan Jupyter Notebook (Opsional)
```bash
jupyter notebook notebooks/
```

## 🚀 Penggunaan

### Metode 1: Menjalankan Semua Analisis Sekaligus
```bash
python main.py
```
Script ini akan menjalankan seluruh pipeline:
1. 📥 Load data
2. 🧹 Preprocessing
3. 📊 EDA (Exploratory Data Analysis)
4. 🔍 Clustering dengan K-Means
5. 📈 Visualisasi hasil
6. 💾 Simpan hasil

### Metode 2: Menjalankan Per Modul
```bash
# Import modul dan jalankan fungsi tertentu
python -c "
from src.data_loader import load_data
from src.preprocessing import preprocess_data

df = load_data('data/raw/Mall_Customers.csv')
df_processed = preprocess_data(df)
print('Data shape:', df_processed.shape)
"
```

### Metode 3: Menggunakan Jupyter Notebook
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Metode 4: Fungsi Individual
```python
# Contoh penggunaan di Python script
import sys
sys.path.append('src')

from data_loader import load_data
from eda import perform_eda
from clustering import perform_clustering

# Load data
df = load_data('data/raw/Mall_Customers.csv')

# Exploratory Analysis
perform_eda(df)

# Clustering
clusters, model = perform_clustering(df, n_clusters=5)
```

## 📊 Hasil Analisis

### Segmentasi Pelanggan yang Ditemukan
Setelah analisis, ditemukan **5 segmen pelanggan**:

| Klaster | Jumlah | Profil | Karakteristik | Strategi Pemasaran |
|---------|--------|--------|---------------|-------------------|
| **Klaster 0** | 39 pelanggan | 🎯 **Pelanggan Premium** | Usia 30-40, Pendapatan tinggi ($86k), Spending tinggi (82) | VIP treatment, produk luxury, personal shopper |
| **Klaster 1** | 35 pelanggan | 💼 **Pelanggan Hemat** | Usia 25-45, Pendapatan tinggi ($87k), Spending rendah (17) | Edukasi produk, diskon eksklusif, program tabungan |
| **Klaster 2** | 23 pelanggan | 👵 **Senior Konservatif** | Usia >55, Pendapatan rendah ($41k), Spending rendah (40) | Diskon hari senior, produk kesehatan, layanan khusus |
| **Klaster 3** | 81 pelanggan | 👫 **Keluarga Menengah** | Semua usia, Pendapatan menengah ($55k), Spending menengah (49) | Paket keluarga, program loyalitas, promo akhir pekan |
| **Klaster 4** | 22 pelanggan | 🎉 **Anak Muda Boros** | Usia <30, Pendapatan rendah ($25k), Spending tinggi (79) | Tren terkini, promo event, pembayaran cicilan |

### Key Insights
1. **Korelasi Negatif** antara usia dan spending score (-0.33)
2. **Tidak ada korelasi signifikan** antara pendapatan dan spending score (0.01)
3. **Segmentasi optimal** adalah 5 klaster berdasarkan metode elbow dan silhouette score
4. **Pelanggan wanita** cenderung memiliki spending score lebih tinggi (53.3 vs 46.8)

## 🔬 Metodologi

### 1. **Exploratory Data Analysis (EDA)**
- Analisis distribusi setiap fitur
- Deteksi outlier dengan boxplot
- Analisis korelasi dengan heatmap
- Visualisasi hubungan antar variabel

### 2. **Preprocessing**
- Encoding variabel kategorikal (Gender → 0/1)
- StandardScaler untuk normalisasi fitur
- Validasi data completeness

### 3. **Clustering dengan K-Means**
- **Penentuan K optimal**: Elbow Method + Silhouette Score
- **Algoritma**: K-Means++ dengan 10 inisialisasi
- **Metrik**: WCSS (Within-Cluster Sum of Squares)

### 4. **Validasi & Interpretasi**
- Analisis centroid tiap klaster
- Visualisasi 2D & 3D
- Profiling berdasarkan statistik deskriptif

## 🛠️ Teknologi

**Bahasa & Framework:**
- ![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)
- ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

**Data Science Stack:**
- ![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?logo=pandas)
- ![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?logo=numpy)
- ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-F7931E?logo=scikit-learn)
- ![SciPy](https://img.shields.io/badge/SciPy-1.7%2B-8CAAE6?logo=scipy)

**Visualisasi:**
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5%2B-11557C?logo=python)
- ![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-5599FF)

## 🤝 Kontribusi

Kontribusi sangat diterima! Ikuti langkah berikut:

1. **Fork** repository ini
2. **Buat branch** baru (`git checkout -b feature/improvement`)
3. **Commit** perubahan (`git commit -m 'Menambahkan fitur X'`)
4. **Push** ke branch (`git push origin feature/improvement`)
5. **Buat Pull Request**

### Guidelines
- Gunakan **Black** untuk formatting code
- Tulis **docstring** untuk fungsi baru
- Tambahkan **unit test** untuk kode baru
- Update **documentation** sesuai perubahan

## 📝 Lisensi

Distribusi di bawah lisensi MIT. Lihat file [LICENSE](LICENSE) untuk detail lebih lanjut.

## 📞 Kontak

**Fitria LM** - [@fitrlm](https://x.com/fitrlm) - fitrialm26@gmail.com

**Link Project:** [https://github.com/fitria-lm/mall-customer-segmentation](https://github.com/fitria-lm/mall-customer-segmentation)

---

## 🙏 Acknowledgments

- Dataset dari [Kaggle](https://www.kaggle.com/datasets/abdallahwagih/mall-customers-segmentation/data)
- Inspirasi dari berbagai tutorial machine learning
- Komunitas data science Indonesia

## ⭐ Support

Jika project ini membantu Anda, berikan ⭐ di GitHub!

---

<div align="center">
  
**Dibuat dengan ❤️ untuk komunitas Data Science**

<sub>Terakhir diperbarui: {tanggal_update}</sub>

</div>

---

## 📚 Referensi

1. [Scikit-learn Documentation - K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
2. [Kaggle Notebook - Customer Segmentation Tutorial](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)
3. [Towards Data Science - Customer Segmentation](https://towardsdatascience.com/customer-segmentation-using-k-means-clustering-d33964f238c3)
4. [Analytics Vidhya - Complete Guide to K-Means Clustering](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/)

## 🔄 Changelog

### v1.0.0 (Current)
- Implementasi K-Means clustering
- Visualisasi EDA lengkap
- Dokumentasi komprehensif
- Unit testing framework

### v0.1.0 (Initial)
- Setup project structure
- Basic data loading & preprocessing
- Exploratory analysis

---
