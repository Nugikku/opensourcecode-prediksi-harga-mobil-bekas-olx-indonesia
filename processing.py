"""
Script Preprocessing Data Mobil Bekas OLX (Versi Terbaru)
Pembaruan:
1. Menangani format rentang jarak_tempuh (misal '65.000-70.000' dirata-rata jadi 67500).
2. Ekstraksi otomatis 4 digit tahun dari 'judul' jika kolom 'tahun' kosong/None.
3. Ekstraksi otomatis transmisi (otomatis/manual) dari teks 'judul'.
4. Pembersihan angka harga dan normalisasi teks menggunakan kamus slang otomotif.
"""

import pandas as pd
import re
import string
from kamus_slang_otomotif import KAMUS_SLANG_OTOMOTIF

INPUT_FILE = "dataset_olx_mentah.csv"
OUTPUT_FILE = "dataset_olx_bersih.csv"

# ============================================================
# 1. TEXT CLEANING & NORMALISASI
# ============================================================

def bersihkan_dan_normalisasi_teks(teks):
    if pd.isna(teks):
        return ""
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+', '', teks)
    teks = re.sub(r'wa\s*\d+|08\d+', 'kontak', teks)
    teks = teks.translate(str.maketrans('', '', string.punctuation))
    teks = re.sub(r'\s+', ' ', teks).strip()

    kata_kata = teks.split()
    kata_terfilter = [KAMUS_SLANG_OTOMOTIF.get(k, k) for k in kata_kata]
    return ' '.join(kata_terfilter)

# ============================================================
# 2. NUMERICAL & FEATURE EXTRACTION
# ============================================================

def bersihkan_angka_harga(nilai):
    """Membersihkan nilai harga baik yang bertipe float/notasi ilmiah maupun string."""
    if pd.isna(nilai):
        return None
    try:
        val_float = float(nilai)
        return int(val_float)
    except (ValueError, TypeError):
        pass
    angka_saja = re.sub(r'\D', '', str(nilai))
    return int(angka_saja) if angka_saja else None

def perbaiki_jarak_tempuh(val):
    """Menangani rentang jarak tempuh OLX (misal: '65.000-70.000' -> dihitung rata-ratanya)."""
    if pd.isna(val):
        return None
    val_str = str(val).replace('.', '').strip()
    if '-' in val_str:
        bagian = val_str.split('-')
        angka = [int(re.sub(r'\D', '', b)) for b in bagian if re.sub(r'\D', '', b)]
        return int(sum(angka) / len(angka)) if angka else None
    angka_saja = re.sub(r'\D', '', val_str)
    return int(angka_saja) if angka_saja else None

def cari_tahun(row):
    """Ambil tahun dari kolom 'tahun'. Jika kosong, cari 4 digit tahun (1990-2026) di judul."""
    val = row['tahun']
    if pd.notna(val) and str(val).strip() not in ['', 'None', 'nan']:
        try:
            return int(float(val))
        except ValueError:
            pass
    
    judul = str(row.get('judul', ''))
    cocok = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', judul)
    if cocok:
        return int(cocok[-1])
    return None

def tentukan_transmisi(row):
    """Deteksi transmisi dari gabungan kolom transmisi, teks judul, dan cuplikan deskripsi."""
    gabungan = f"{row.get('transmisi', '')} {row.get('judul', '')} {str(row.get('deskripsi', ''))[:300]}".lower()
    if any(k in gabungan for k in [' matic', '-at', ' at ', ' a/t', 'otomatis', ' tiptronic', ' triptonic', ' cvt ']):
        return 'otomatis'
    elif any(k in gabungan for k in [' manual', '-mt', ' mt ', ' m/t']):
        return 'manual'
    return 'manual'

# ===========================================================
# 3. PIPELINE UTAMA
# ===========================================================

def main():
    print("Membaca file data mentah OLX...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"[error] File '{INPUT_FILE}' tidak ditemukan.")
        return

    print(f"Data awal: {len(df)} baris")
    
    # 1. Ekstraksi Tahun & Transmisi
    df['tahun'] = df.apply(cari_tahun, axis=1)
    df['transmisi'] = df.apply(tentukan_transmisi, axis=1)

    # 2. Pembersihan Angka (Harga & Jarak Tempuh)
    df['harga'] = df['harga'].apply(bersihkan_angka_harga)
    df['jarak_tempuh'] = df['jarak_tempuh'].apply(perbaiki_jarak_tempuh)

    # 3. Filter Data Masuk Akal & Validitas Tahun & Harga
    df = df.dropna(subset=['harga', 'tahun'])
    df = df[(df['harga'] >= 25_000_000) & (df['harga'] <= 2_500_000_000)]
    df = df[(df['tahun'] >= 1995) & (df['tahun'] <= 2026)]
    df['tahun'] = df['tahun'].astype(int)

    # 4. Standarisasi Format Merek (Title Case)
    df['merek'] = df['merek'].astype(str).str.strip().str.title()

    # 5. Hapus Duplikat
    if 'id_iklan' in df.columns:
        df = df.drop_duplicates(subset=['id_iklan'], keep='first')
    df = df.drop_duplicates(subset=['merek', 'judul', 'tahun', 'jarak_tempuh', 'harga'], keep='first')

    # 6. Tangani Missing Values pada Jarak Tempuh dengan Median per Tahun (Fallback Median Global)
    median_global = df['jarak_tempuh'].dropna().median()
    df['jarak_tempuh'] = df.groupby('tahun')['jarak_tempuh'].transform(lambda s: s.fillna(s.median()))
    df['jarak_tempuh'] = df['jarak_tempuh'].fillna(median_global).astype(int)

    # 7. Normalisasi Kolom Teks
    if 'deskripsi' in df.columns:
        df['deskripsi_bersih'] = df['deskripsi'].apply(bersihkan_dan_normalisasi_teks)
    if 'judul' in df.columns:
        df['judul_bersih'] = df['judul'].apply(bersihkan_dan_normalisasi_teks)

    df = df.reset_index(drop=True)
    print(f"Data bersih siap latih: {len(df)} baris")

    # 8. Simpan Hasil
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"Selesai! Disimpan ke '{OUTPUT_FILE}'\n")
    print("Contoh 5 data teratas:")
    print(df[['merek', 'tahun', 'transmisi', 'jarak_tempuh', 'harga']].head())

if __name__ == "__main__":
    main()