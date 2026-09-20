# 🚗 AutoValuate: Prediksi Harga Mobil Bekas OLX Indonesia

Proyek Data Mining komparatif untuk memprediksi harga pasar wajar mobil bekas menggunakan data riil hasil web scraping dari **OLX Indonesia**. Model machine learning dilatih dan dikomparasikan menggunakan **5 paradigma algoritma yang berbeda**: *Ridge Regression (Baseline)*, *K-Nearest Neighbors (KNN)*, *Decision Tree*, *Gradient Boosting*, dan *Random Forest Regressor*.

Model terbaik dilengkapi dengan fitur interaktif **AutoValuate (Smart Deal Detector & Negotiation Support)** untuk memandu pembeli dalam menilai kewajaran harga iklan dan memberikan batas tawar-menawar beserta skrip negosiasi siap pakai.

---

## 🏆 Hasil Komparasi Performa 5 Model Machine Learning

Berdasarkan evaluasi pengujian data riil (split 80% train : 20% test):

| Algoritma / Model | Paradigma Pembelajaran | $R^2$ Train | $R^2$ Test (Akurasi) | MAE (Rata-rata Error) | Keterangan |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | *Ensemble Bagging* | 0.9693 | **0.8846** | **Rp 61.800.458** | 🏆 **Model Terbaik (Pemenang)** |
| **Gradient Boosting Regressor** | *Ensemble Boosting* | 0.8428 | **0.8395** | Rp 79.976.718 | Performa sangat stabil |
| **Ridge Regression** | *Parametrik Linier (Baseline)* | 0.7359 | **0.7752** | Rp 91.838.255 | Baseline linier terstandardisasi |
| **K-Nearest Neighbors (KNN)** | *Instance-Based (Jarak)* | 0.8315 | **0.7465** | Rp 78.233.580 | Berbasis kemiripan spesifikasi |
| **Decision Tree Regressor** | *Pohon Keputusan Tunggal* | 0.9952 | **0.7452** | Rp 77.521.462 | Terindikasi *overfitting* |

> 📌 **Temuan Data Mining:** *Decision Tree Tunggal* mengalami *overfitting* ekstrem ($R^2$ data latih 0.9952 vs data uji 0.7452). Pendekatan *Ensemble Bagging* pada **Random Forest** berhasil mengatasi kelemahan tersebut dan mendongkrak akurasi hingga **$R^2 = 0.8846$** dengan rata-rata deviasi error terendah.

---

## 📁 Struktur File Repositori

```text
scripts_prediksi/
│
├── scraping.py                          # Scraping data iklan mobil OLX via Playwright
├── kamus_slang_otomotif.py              # Kamus normalisasi teks slang otomotif
├── processing.py                        # Preprocessing, imputasi & pembersihan data
├── train_model.py                       # Training & komparasi 5 model ML + visualisasi
├── test_prediksi.py                     # AutoValuate (Deal detector & panduan negosiasi)
│
├── dataset_olx_mentah.csv               # Output scraping (data mentah ~3000 data)
├── dataset_olx_bersih.csv               # Output preprocessing (data bersih siap latih)
├── model_prediksi_mobil.pkl             # Model Random Forest terbaik tersimpan
└── output_gambar_pelatihan_DD-MM-YYYY/  # Grafik visualisasi & ringkasan metrik CSV
    ├── perbandingan_r2.png
    ├── perbandingan_error.png
    ├── aktual_vs_prediksi.png
    ├── residual_plot.png
    ├── feature_importance.png
    └── ringkasan_metrik.csv
```

---

## ⚙️ Instalasi & Persiapan

### 1. Clone Repositori
```bash
git clone https://github.com/Nugikku/opensourcecode-prediksi-harga-mobil-bekas-olx-indonesia.git
cd opensourcecode-prediksi-harga-mobil-bekas-olx-indonesia
```

### 2. Install Library Python yang Dibutuhkan
```bash
pip install pandas numpy scikit-learn matplotlib playwright
```

### 3. Install Browser Chromium untuk Playwright (Khusus Scraping)
```bash
playwright install chromium
```

---

## 🔄 Alur & Urutan Eksekusi

```text
OLX.co.id
    │
    ▼
[scraping.py] ──────────────► dataset_olx_mentah.csv
                                        │
                                        ▼
                             [processing.py] ─────────► dataset_olx_bersih.csv
                                                                  │
                                                                  ▼
                                                       [train_model.py] ──────► model_prediksi_mobil.pkl
                                                                                           │
                                                                                           ▼
                                                                               [test_prediksi.py]
```

### Langkah 1 — Web Scraping Data OLX (`scraping.py`)
Mengambil data listing iklan mobil bekas secara otomatis dengan teknik intersepsi API JSON browser:
```bash
python scraping.py
```
* **Output:** `dataset_olx_mentah.csv` (target default 3.000 data iklan).
* **Kolom Mentah:** `id_iklan`, `judul`, `deskripsi`, `merek`, `model`, `tahun`, `transmisi`, `jarak_tempuh`, `harga`.

---

### Langkah 2 — Preprocessing & Pembersihan Data (`processing.py`)
Membersihkan data mentah agar valid, konsisten, dan siap dimodelkan:
```bash
python processing.py
```
* **Output:** `dataset_olx_bersih.csv`.
* **Tahapan Preprocessing:**
  - ✅ Ekstraksi otomatis 4-digit tahun dari teks judul jika kolom tahun kosong.
  - ✅ Deteksi transmisi cerdas (manual / otomatis) dari gabungan teks `judul` dan `deskripsi`.
  - ✅ Pembersihan nilai numerik harga dan jarak tempuh (termasuk format rentang seperti `65.000-70.000` dirata-rata).
  - ✅ Standarisasi kapitalisasi merek (*Title Case*).
  - ✅ Deduplikasi data berdasarkan `id_iklan` dan kombinasi spesifikasi.
  - ✅ **Imputasi Jarak Tempuh Realistis:** Mengisi missing value kilometer menggunakan nilai median per kelompok tahun pembuatan mobil (`df.groupby('tahun')`).
  - ✅ Filter outlier masuk akal (rentang harga Rp 25 Juta – Rp 2,5 Miliar; tahun 1995 – 2026).

---

### Langkah 3 — Pelatihan & Komparasi 5 Model (`train_model.py`)
Melatih dan membandingkan 5 model Machine Learning secara adil tanpa kebocoran data (*data leakage*):
```bash
python train_model.py
```
* **Fitur Utama (X):** `merek`, `model` (seri mobil), `tahun`, `transmisi`, `jarak_tempuh`.
* **Target (Y):** `harga`.
* **Arsitektur Pipeline:**
  - Kolom Kategorikal di-*encode* dengan `OneHotEncoder(handle_unknown='ignore')`.
  - Kolom Numerik dinormalisasi dengan `StandardScaler()`.
  - Dibungkus rapi dalam `Pipeline` scikit-learn sehingga bebas kebocoran data uji.
* **Output:**
  - File model terbaik: `model_prediksi_mobil.pkl`.
  - Folder bertanggal `output_gambar_pelatihan_DD-MM-YYYY/` berisi 5 grafik visualisasi (Bar chart $R^2$, Bar chart MAE/RMSE, Scatter Aktual vs Prediksi, Analisis Residual, dan 15 Fitur Paling Berpengaruh) serta file `ringkasan_metrik.csv`.

---

### Langkah 4 — Pengujian Interaktif & Deteksi Penawaran (`test_prediksi.py`)
Menjalankan sistem prediksi interaktif **AutoValuate**:
```bash
python test_prediksi.py
```

**Fitur yang Disediakan:**
1. **Ringkasan Komparasi Otomatis:** Menampilkan metrik performa 5 model terbaru secara dinamis dari file CSV pelatihan.
2. **Dropdown Pemilihan Kendaraan:** Memilih merek dan seri mobil yang tersedia secara terstruktur.
3. **Smart Deal Detector:**
   * `[✓] HARGA WAJAR PASAR (FAIR DEAL)` jika harga iklan berada di rentang wajar model.
   * `[!] KEMAHALAN / OVERPRICED` jika harga iklan di atas estimasi wajar.
   * `[?] TERLALU MURAH / WASPADA RIWAYAT UNIT` jika harga penawaran jauh di bawah harga pasaran (peringatan cek BPKB/bekas tabrak/banjir).
4. **Deal Score (0–100):** Skor kelayakan harga unit.
5. **Panduan Batas Tawar-Menawar:**
   * Tawaran Awal (*Opening Bid*)
   * Batas Maksimal Kesepakatan
6. **Skrip Argumen Negosiasi Siap Pakai:** Template percakapan yang siap di-copy-paste pembeli saat bernegosiasi dengan penjual di chat OLX atau WhatsApp.
7. **Pengujian 5 Sampel Data Riil:** Menampilkan perbandingan harga asli iklan vs harga prediksi model beserta nominal selisihnya.

---

## 🛠️ Teknologi & Library

* **Python 3.x**
* **Pandas & NumPy** — Manipulasi, pembersihan, dan kalkulasi array data.
* **Scikit-Learn** — Pipeline machine learning, preprocessing, ColumnTransformer, serta pemodelan regresi.
* **Matplotlib** — Pembuatan visualisasi dan grafik evaluasi pelatihan.
* **Playwright** — Otomatisasi browser Chromium untuk web scraping dinamis.

---

## ⚠️ Disclaimer

Dataset yang digunakan dalam proyek ini diperoleh melalui web scraping dari OLX Indonesia semata-mata untuk keperluan **akademis, penelitian, dan edukasi** (mata kuliah Data Mining). Proyek ini tidak berafiliasi secara resmi dengan OLX dan tidak dipergunakan untuk tujuan komersial.