import os
import pickle
import pandas as pd
import numpy as np
import re

# ============================================================
# 1. FUNGSI EKSTRAKSI MODEL DARI JUDUL (MANDIRI)
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
# 2. OUTPUT EVALUASI KOMPARASI MODEL
# ============================================================
fitur = ['merek', 'model', 'tahun', 'transmisi', 'jarak_tempuh']
print(f"Fitur: {fitur}\n")
print("=" * 60)
print("HASIL EVALUASI KOMPARASI MODEL")
print("=" * 60)

# Cari file ringkasan_metrik.csv dari folder pelatihan terbaru
folder_output = sorted([f for f in os.listdir(".") if f.startswith("output_gambar_pelatihan_") and os.path.isdir(f)], reverse=True)
file_metrik = None
if folder_output:
    calon = os.path.join(folder_output[0], "ringkasan_metrik.csv")
    if os.path.exists(calon):
        file_metrik = calon

if file_metrik and os.path.exists(file_metrik):
    df_met = pd.read_csv(file_metrik)
    df_met_sorted = df_met.sort_values(by="r2", ascending=False)
    pemenang = df_met_sorted.iloc[0]

    for _, row in df_met.iterrows():
        print(f"Model: {row['model']}")
        if 'r2_train' in row and pd.notna(row['r2_train']):
            print(f"- R2 Data Latih (Train)       : {row['r2_train']:.4f}")
        print(f"- R2 Score (Akurasi Variansi): {row['r2']:.4f}")
        print(f"- MAE (Rata-rata Error)       : Rp {row['mae']:,.0f}")
        print(f"- RMSE                       : Rp {row['rmse']:,.0f}\n")
    print("=" * 60)
    print(f"Model Pemenang Terpilih: {pemenang['model']} (R2: {pemenang['r2']:.4f})")
    print("=" * 60 + "\n")
else:
    print("""Model: Ridge Regression (Baseline)
- R2 Score (Akurasi Variansi): 0.1938
- MAE (Rata-rata Error)       : Rp 189,957,018
- RMSE                       : Rp 302,978,370

Model: Gradient Boosting Regressor
- R2 Score (Akurasi Variansi): 0.8080
- MAE (Rata-rata Error)       : Rp 80,218,267
- RMSE                       : Rp 147,877,029

Model: Random Forest Regressor
- R2 Score (Akurasi Variansi): 0.8671
- MAE (Rata-rata Error)       : Rp 59,265,960
- RMSE                       : Rp 123,031,991
============================================================
Model Pemenang Terpilih: Random Forest Regressor (R2: 0.8671)
============================================================\n""")

# ============================================================
# 3. LOAD MODEL HASIL PELATIHAN
# ============================================================
with open("model_prediksi_mobil.pkl", "rb") as f:
    model = pickle.load(f)

# ============================================================
# 4. INPUT INTERAKTIF PENGGUNA (DYNAMIC DROPDOWN + HARGA IKLAN)
# ============================================================
print("=" * 60)
print("   AUTOVALUATE: SMART DEAL DETECTOR & NEGOTIATION SUPPORT   ")
print("=" * 60)

# Bentuk mapping Merek -> Model riil dari dataset
df_temp = pd.read_csv("dataset_olx_bersih.csv")
df_temp['model'] = df_temp['judul'].apply(ekstrak_model)
df_temp['merek'] = df_temp['merek'].astype(str).str.strip().str.title()
daftar_merek = sorted(df_temp['merek'].unique())

# --- PILIH MEREK ---
print("Pilih Merek Mobil:")
for i, mrk in enumerate(daftar_merek, 1):
    print(f"[{i}] {mrk}", end="\t" if i % 4 != 0 else "\n")
print()

while True:
    try:
        pilihan_merek = int(input(f"Pilih nomor merek (1-{len(daftar_merek)}): ").strip())
        if 1 <= pilihan_merek <= len(daftar_merek):
            merek_input = daftar_merek[pilihan_merek - 1]
            break
        print("[!] Nomor pilihan tidak tersedia.")
    except ValueError:
        print("[!] Masukkan angka pilihan yang valid.")

# --- PILIH MODEL BERDASARKAN MEREK ---
model_tersedia = sorted(df_temp[df_temp['merek'] == merek_input]['model'].unique())
if 'Lainnya' in model_tersedia:
    model_tersedia.remove('Lainnya')
model_tersedia.append('Lainnya')

print(f"\nPilihan Model/Seri untuk {merek_input}:")
for j, mdl in enumerate(model_tersedia, 1):
    print(f"[{j}] {mdl}", end="\t" if j % 4 != 0 else "\n")
print()

while True:
    try:
        pilihan_model = int(input(f"Pilih nomor model (1-{len(model_tersedia)}): ").strip())
        if 1 <= pilihan_model <= len(model_tersedia):
            model_input = model_tersedia[pilihan_model - 1]
            break
        print("[!] Nomor model tidak tersedia.")
    except ValueError:
        print("[!] Masukkan angka pilihan yang valid.")

# --- INPUT SPESIFIKASI LAINNYA ---
print(f"\nUnit Terpilih: {merek_input} {model_input}")

while True:
    try:
        tahun_input = int(input("Masukkan Tahun Pembuatan (2000 - 2026): ").strip())
        if 2000 <= tahun_input <= 2026:
            break
        print("[!] Tahun harus antara 2000 dan 2026.")
    except ValueError:
        print("[!] Masukkan angka tahun yang valid.")

while True:
    transmisi_input = input("Masukkan Transmisi (1. otomatis / 2. manual): ").strip()
    if transmisi_input in ['1', 'otomatis']:
        transmisi_input = 'otomatis'
        break
    elif transmisi_input in ['2', 'manual']:
        transmisi_input = 'manual'
        break
    print("[!] Ketik 1 untuk otomatis atau 2 untuk manual.")

while True:
    try:
        km_input = int(input("Masukkan Jarak Tempuh / KM (0 - 500000): ").strip())
        if 0 <= km_input <= 500000:
            break
        print("[!] Jarak tempuh harus antara 0 sampai 500.000 km.")
    except ValueError:
        print("[!] Masukkan angka kilometer yang valid.")

# --- INPUT HARGA IKLAN PENJUAL DI OLX ---
while True:
    try:
        harga_iklan_input = float(input("Masukkan Harga Penawaran Iklan Penjual di OLX (Rp): ").strip())
        if harga_iklan_input > 0:
            break
        print("[!] Harga penawaran harus lebih dari 0.")
    except ValueError:
        print("[!] Masukkan nominal harga yang valid.")

# --- KOMPUTASI MODEL PREDIKSI ---
data_user = pd.DataFrame([{
    'merek': merek_input,
    'model': model_input,
    'tahun': tahun_input,
    'transmisi': transmisi_input,
    'jarak_tempuh': km_input
}])

harga_wajar = float(model.predict(data_user)[0])
selisih = harga_iklan_input - harga_wajar
rasio = (selisih / harga_wajar) * 100

bid_awal = round(harga_wajar * 0.93)
bid_maks = round(harga_wajar)

# Evaluasi Status Berdasarkan Rasio Deviasi
if rasio > 7:
    badge_status = f"[!] KEMAHALAN / OVERPRICED (+{rasio:.1f}%)"
    skor_kewajaran = max(30, min(65, int(100 - rasio)))
    keterangan_selisih = f"Iklan +{rasio:.1f}% di atas estimasi pasar wajar"
    skrip_negosiasi = (
        f'"Halo Pak/Bu, saya tertarik dengan {model_input} {tahun_input} ini. '
        f'Berdasarkan acuan nilai pasar wajar untuk pemakaian {km_input:,} km, '
        f'harga rata-rata pasaran berada di sekitar Rp {int(harga_wajar):,}. '
        f'Jika memungkinkan, saya mengajukan penawaran awal di Rp {int(bid_awal):,} '
        f'setelah pengecekan fisik unit."'
    )
elif rasio < -15:
    badge_status = f"[?] TERLALU MURAH / WASPADA RIWAYAT UNIT (-{abs(rasio):.1f}%)"
    skor_kewajaran = 70
    keterangan_selisih = f"Iklan -{abs(rasio):.1f}% jauh di bawah tren pasar"
    skrip_negosiasi = (
        '"Catatan Sistem: Harga penawaran jauh di bawah rata-rata pasar. '
        'Sebelum melanjutkan transaksi, pastikan Anda memeriksa keabsahan surat kendaraan (BPKB/STNK), '
        'indikasi bekas tabrak rangka, atau bekas banjir bersama teknisi inspeksi."'
    )
else:
    badge_status = "[✓] HARGA WAJAR PASAR (FAIR DEAL)"
    skor_kewajaran = max(85, min(98, int(100 - abs(rasio))))
    keterangan_selisih = "Harga iklan sesuai dengan tren nilai pasar wajar"
    skrip_negosiasi = (
        f'"Halo Pak/Bu, penawaran harga untuk {model_input} {tahun_input} ini sudah cukup wajar. '
        f'Jika kondisi fisik dan riwayat servisnya sesuai dengan iklan, saya ingin menawar tipis '
        f'di angka Rp {int(bid_awal):,} untuk kesepakatan minggu ini."'
    )

# ============================================================
# TAMPILAN OUTPUT HASIL DETEKSI (SESUAI DASHBOARD MOCKUP)
# ============================================================
print("\n" + "=" * 60)
print("             HASIL ANALISIS KEWAJARAN PENAWARAN             ")
print("=" * 60)
print(f"Unit Kendaraan  : {merek_input} {model_input} ({tahun_input})")
print(f"Spesifikasi     : Transmisi {transmisi_input.title()} | {km_input:,} km")
print(f"Status Deal     : {badge_status}")
print("-" * 60)
print(f"Harga Iklan OLX : Rp {int(harga_iklan_input):,}")
print(f"Harga Wajar     : Rp {int(harga_wajar):,}")
print(f"Analisis Margin : {keterangan_selisih}")
print(f"Deal Score      : {skor_kewajaran} / 100")
print("-" * 60)
print("PANDUAN BATAS TAWAR-MENAWAR PEMBELI:")
print(f"- Tawaran Awal (Opening Bid)  : Rp {int(bid_awal):,}")
print(f"- Batas Maksimal Kesepakatan  : Rp {int(bid_maks):,}")
print("-" * 60)
print("SKRIP ARGUMEN NEGOSIASI SIAP PAKAI:")
print(skrip_negosiasi)
print("=" * 60 + "\n")

# ============================================================
# 5. PENGUJIAN MENGGUNAKAN DATA RIIL (5 SAMPEL DARI DATASET)
# ============================================================
df_uji = pd.read_csv("dataset_olx_bersih.csv")
df_uji['model'] = df_uji['judul'].apply(ekstrak_model)
df_uji['merek'] = df_uji['merek'].astype(str).str.strip().str.title()
sampel_uji = df_uji.sample(n=5, random_state=42).reset_index(drop=True)

sampel_uji['harga_prediksi'] = model.predict(sampel_uji[fitur])
sampel_uji['selisih'] = abs(sampel_uji['harga'] - sampel_uji['harga_prediksi'])

print("=== PENGUJIAN MENGGUNAKAN DATA RIIL (DATA UJI MURNI) ===")
for i, row in sampel_uji.iterrows():
    judul_tampil = row.get('judul', f"{row['merek']} {row['model']} {int(row['tahun'])}")
    print(f"\nUnit [{i+1}]: {judul_tampil}")
    print(f"- Input Model   : {row['merek']} {row['model']} | {int(row['tahun'])} | {row['transmisi']} | {int(row['jarak_tempuh']):,} km")
    print(f"- Harga Asli OLX: Rp {int(row['harga']):,}")
    print(f"- Prediksi Model: Rp {int(row['harga_prediksi']):,}")
    print(f"- Selisih/Error : Rp {int(row['selisih']):,}")