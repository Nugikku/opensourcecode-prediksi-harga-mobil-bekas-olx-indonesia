"""
Script Training Model Prediksi Harga Mobil OLX (Versi Lengkap)
Alur:
1. Load dataset bersih
2. Ekstraksi tipe/seri mobil (model) dari teks 'judul'
3. Split train-test (80:20)
4. Pipeline Preprocessing (OneHotEncoder untuk kategorikal, Passthrough untuk numerik)
5. Komparasi performa model regresi:
   - Linear / Ridge Regression (Baseline)
   - Gradient Boosting Regressor
   - Random Forest Regressor
6. Simpan model terbaik ke 'model_prediksi_mobil.pkl'
7. Simpan visualisasi (PNG) ke folder 'output_gambar/':
   - perbandingan_r2.png      -> bar chart R2 antar model
   - perbandingan_error.png   -> bar chart MAE & RMSE antar model
   - aktual_vs_prediksi.png   -> scatter plot aktual vs prediksi (model terbaik)
   - residual_plot.png        -> distribusi residual (model terbaik)
   - feature_importance.png   -> pentingnya fitur (khusus model tree-based)
   - ringkasan_metrik.csv     -> tabel metrik semua model (untuk laporan)
"""

import os
import pandas as pd
import numpy as np
import pickle
import re
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # render ke file, tidak butuh display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Folder output dibuat baru setiap kali training dijalankan, ditandai
# tanggal jalannya (format: output_gambar_pelatihan_DD-MM-YYYY).
# Jadi hasil training hari ini tidak menimpa hasil training sebelumnya.
TANGGAL_HARI_INI = datetime.now().strftime("%d-%m-%Y")
OUTPUT_DIR = f"output_gambar_pelatihan_{TANGGAL_HARI_INI}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# 1. DAFTAR SERI MOBIL UNTUK EKSTRAKSI DARI JUDUL
# ============================================================
DAFTAR_MODEL = [
    # Toyota
    "alphard", "vellfire", "avanza", "innova", "fortuner", "yaris", "rush",
    "calya", "agya", "raize", "corolla", "camry", "vios", "hilux", "sienta",
    "granace", "land cruiser", "hiace", "veloz", "harrier",
    # Honda
    "brio", "hrv", "crv", "city", "civic", "jazz", "mobilio", "brv", "accord", "freed", "odyssey", "wrv",
    # Daihatsu
    "xenia", "sigra", "terios", "ayla", "rocky", "sirion", "gran max", "luxio",
    # Mitsubishi
    "xpander", "pajero", "outlander", "mirage", "triton", "eclipse",
    # Suzuki
    "ertiga", "xl7", "ignis", "baleno", "jimny", "karimun", "sx4", "s-presso", "grand vitara", "every",
    # Nissan
    "grand livina", "livina", "serena", "xtrail", "juke", "march", "kicks", "magnite", "teana", "elgrand",
    # BMW & Mercedes
    "320i", "330i", "520i", "530i", "x1", "x3", "x5", "x7",
    "c200", "c300", "e200", "e250", "e300", "s450", "glc", "gla", "gle", "amg", "cla",
    # Jeep & Mini
    "rubicon", "wrangler", "sahara", "cherokee", "compass", "renegade",
    "cooper", "countryman", "clubman",
    # Hyundai & Wuling
    "creta", "stargazer", "santa fe", "palisade", "ioniq", "tucson", "h-1",
    "confero", "almaz", "cortez", "air ev", "binguo", "alvez",
    # Lainnya
    "bj40", "sealion", "defender", "rx300", "everest", "ranger"
]


def ekstrak_model(judul):
    judul_lower = str(judul).lower()
    for m in DAFTAR_MODEL:
        if re.search(r'\b' + re.escape(m) + r'\b', judul_lower):
            return m.title()
    return "Lainnya"


# ============================================================
# 2. LOAD DATA & PERSIAPAN FITUR
# ============================================================
print("Membaca data bersih...")
file_dataset = "dataset_olx_bersih.csv"
if not os.path.exists(file_dataset):
    for fldr in [f for f in os.listdir(".") if f.startswith("dataset_") and os.path.isdir(f)]:
        calon = os.path.join(fldr, "dataset_olx_bersih.csv")
        if os.path.exists(calon):
            file_dataset = calon
            break

df = pd.read_csv(file_dataset)

# Standarisasi kolom merek & ekstraksi model
df['merek'] = df['merek'].astype(str).str.strip().str.title()
df['model'] = df['judul'].apply(ekstrak_model)

# Atribut yang digunakan untuk memprediksi harga (Sesuai Proposal)
fitur = ['merek', 'model', 'tahun', 'transmisi', 'jarak_tempuh']
target = 'harga'

df_model = df.dropna(subset=fitur + [target]).copy()

X = df_model[fitur]
y = df_model[target]

print(f"Total data siap training: {len(df_model)} baris")
print(f"Fitur: {fitur}")

# Split 80% data latih dan 20% data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# 3. PIPELINE PREPROCESSING
# ============================================================
categorical_cols = ['merek', 'model', 'transmisi']
numerical_cols = ['tahun', 'jarak_tempuh']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)

# ============================================================
# 4. TRAINING & EVALUASI KOMPARASI MODEL
# ============================================================
models = {
    "Ridge Regression (Baseline)": Ridge(),
    "K-Nearest Neighbors (KNN)": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42)
}

print("\n" + "=" * 50)
print("HASIL EVALUASI KOMPARASI MODEL")
print("=" * 50)

best_model_name = None
best_model_pipeline = None
best_r2 = -np.inf
best_y_pred = None

hasil_metrik = []          # untuk tabel/CSV ringkasan
semua_prediksi = {}        # nama_model -> y_pred (dipakai untuk plot pembanding opsional)

for nama_model, regressor in models.items():
    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('reg', regressor)
    ])

    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_pred = pipeline.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nModel: {nama_model}")
    print(f"- R2 Data Latih (Train)       : {r2_train:.4f}")
    print(f"- R2 Data Uji (Test / Akurasi): {r2_test:.4f}")
    print(f"- MAE (Rata-rata Error)       : Rp {mae:,.0f}")
    print(f"- RMSE                       : Rp {rmse:,.0f}")

    hasil_metrik.append({
        "model": nama_model,
        "r2_train": r2_train,
        "r2": r2_test,
        "mae": mae,
        "rmse": rmse,
    })
    semua_prediksi[nama_model] = y_pred

    if r2_test > best_r2:
        best_r2 = r2_test
        best_model_name = nama_model
        best_model_pipeline = pipeline
        best_y_pred = y_pred

print("\n" + "=" * 50)
print(f"Model Pemenang Terpilih: {best_model_name} (R2: {best_r2:.4f})")
print("=" * 50)

df_metrik = pd.DataFrame(hasil_metrik)

# ============================================================
# 5. SIMPAN MODEL FINAL
# ============================================================
with open("model_prediksi_mobil.pkl", "wb") as f:
    pickle.dump(best_model_pipeline, f)

print("\nModel terbaik berhasil disimpan ke 'model_prediksi_mobil.pkl'!")

# ============================================================
# 6. VISUALISASI HASIL (DISIMPAN SEBAGAI GAMBAR)
# ============================================================
print(f"\nMembuat visualisasi ke folder '{OUTPUT_DIR}/'...")

warna_model = {
    "Ridge Regression (Baseline)": "#9CA3AF",
    "K-Nearest Neighbors (KNN)": "#F59E0B",
    "Decision Tree Regressor": "#EC4899",
    "Gradient Boosting Regressor": "#60A5FA",
    "Random Forest Regressor": "#34D399",
}


def _tandai_pemenang(labels):
    """Beri label dengan bintang untuk model pemenang."""
    return [f"{lbl} *" if lbl == best_model_name else lbl for lbl in labels]


# --- 6.1 Bar chart perbandingan R2 ---
fig, ax = plt.subplots(figsize=(10, 5.5))
warna = [warna_model.get(m, "#9CA3AF") for m in df_metrik["model"]]
bars = ax.bar(_tandai_pemenang(df_metrik["model"]), df_metrik["r2"], color=warna)
ax.set_ylabel("R2 Score")
ax.set_title("Perbandingan R2 Score Antar Model\n(* = model pemenang)")
ax.set_ylim(0, max(1.0, df_metrik["r2"].max() * 1.15))
for bar, val in zip(bars, df_metrik["r2"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}",
             ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "perbandingan_r2.png"), dpi=150)
plt.close(fig)

# --- 6.2 Bar chart perbandingan MAE & RMSE ---
fig, ax = plt.subplots(figsize=(10.5, 5.5))
x = np.arange(len(df_metrik))
lebar = 0.35
ax.bar(x - lebar / 2, df_metrik["mae"], lebar, label="MAE", color="#F59E0B")
ax.bar(x + lebar / 2, df_metrik["rmse"], lebar, label="RMSE", color="#EF4444")
ax.set_xticks(x)
ax.set_xticklabels(_tandai_pemenang(df_metrik["model"]), rotation=20, ha="right")
ax.set_ylabel("Error (Rp)")
ax.set_title("Perbandingan MAE & RMSE Antar Model\n(* = model pemenang, makin kecil makin baik)")
ax.legend()
ax.get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda val, pos: f"{val:,.0f}")
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "perbandingan_error.png"), dpi=150)
plt.close(fig)

# --- 6.3 Scatter aktual vs prediksi (model terbaik) ---
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(y_test, best_y_pred, alpha=0.4, color="#34D399", edgecolor="none")
batas_min = min(y_test.min(), best_y_pred.min())
batas_max = max(y_test.max(), best_y_pred.max())
ax.plot([batas_min, batas_max], [batas_min, batas_max], "--", color="#374151", linewidth=1.5, label="Prediksi Sempurna")
ax.set_xlabel("Harga Aktual (Rp)")
ax.set_ylabel("Harga Prediksi (Rp)")
ax.set_title(f"Aktual vs Prediksi — {best_model_name}\nR2 = {best_r2:.4f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "aktual_vs_prediksi.png"), dpi=150)
plt.close(fig)

# --- 6.4 Distribusi residual (model terbaik) ---
residual = y_test.values - best_y_pred
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].hist(residual, bins=30, color="#60A5FA", edgecolor="white")
axes[0].axvline(0, color="#374151", linestyle="--")
axes[0].set_title("Distribusi Residual")
axes[0].set_xlabel("Residual (Aktual - Prediksi, Rp)")
axes[0].set_ylabel("Frekuensi")

axes[1].scatter(best_y_pred, residual, alpha=0.4, color="#F59E0B", edgecolor="none")
axes[1].axhline(0, color="#374151", linestyle="--")
axes[1].set_title("Residual vs Prediksi")
axes[1].set_xlabel("Harga Prediksi (Rp)")
axes[1].set_ylabel("Residual (Rp)")

fig.suptitle(f"Analisis Residual — {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_plot.png"), dpi=150)
plt.close(fig)

# --- 6.5 Feature importance (khusus model tree-based: RF / Gradient Boosting) ---
reg_terbaik = best_model_pipeline.named_steps["reg"]
if hasattr(reg_terbaik, "feature_importances_"):
    nama_fitur_encoded = best_model_pipeline.named_steps["prep"].get_feature_names_out()
    importances = reg_terbaik.feature_importances_

    df_importance = pd.DataFrame({
        "fitur": nama_fitur_encoded,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df_importance["fitur"][::-1], df_importance["importance"][::-1], color="#8B5CF6")
    ax.set_xlabel("Tingkat Kepentingan (Importance)")
    ax.set_title(f"15 Fitur Paling Berpengaruh — {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)
else:
    print(f"(Model {best_model_name} tidak punya feature_importances_, grafik ini dilewati)")

# --- 6.6 Simpan tabel metrik sebagai CSV (untuk lampiran laporan) ---
df_metrik.to_csv(os.path.join(OUTPUT_DIR, "ringkasan_metrik.csv"), index=False)

print(f"Visualisasi selesai. Cek folder '{OUTPUT_DIR}/' untuk file-file berikut:")
print("  - perbandingan_r2.png")
print("  - perbandingan_error.png")
print("  - aktual_vs_prediksi.png")
print("  - residual_plot.png")
print("  - feature_importance.png (jika model mendukung)")
print("  - ringkasan_metrik.csv")
print(f"\n(Folder ini khusus untuk pelatihan tanggal {TANGGAL_HARI_INI};")
print(" jalankan ulang script di hari lain akan membuat folder baru.)")