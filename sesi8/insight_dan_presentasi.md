# Final Project — Customer Churn Intelligence Dashboard

> **Draf.** Semua angka di dokumen ini sudah dihitung ulang dari data dan dashboard di repo ini.
> Ubah kalimatnya menjadi kata-kata Anda sendiri sebelum presentasi; bagian penilaian
> "Presentation & Business Insight" menilai interpretasi Anda, bukan teks yang dihafal.

Aplikasi: `sesi8/app_final_dashboard.py` · jalankan `python sesi8/app_final_dashboard.py` → http://127.0.0.1:8053

---

## Bagian 1 — Template final project (§8.9)

### 1. Business Problem
Sebuah penyedia layanan internet kehilangan sekitar satu dari tiga pelanggannya (churn rate 34,9%).
**Pertanyaan bisnis:** pelanggan mana yang berisiko churn, di segmen mana risiko terkonsentrasi,
dan tindakan apa yang perlu diambil?

### 2. Data Understanding
- Sumber: `data/customer_churn.csv`, data latihan sintetis (seed tetap).
- Ukuran: 3.000 pelanggan × 13 kolom; tanggal daftar 1 Jan 2023 – 30 Des 2025.
- Target: `churn` (1 = berhenti berlangganan); churn rate keseluruhan 34,9%.
- Fitur: 5 numerik (tenure, tagihan bulanan, total tagihan, jumlah keluhan, panggilan CS) dan
  5 kategorik (region, segmen, jenis kontrak, metode pembayaran, layanan internet).
- Kualitas data: 45 nilai `total_charges` kosong; tidak ada duplikat `customer_id`.
  Di dashboard final, nilai kosong diisi `monthly_charges × tenure_months` (di EDA Sesi 2 diisi median).

### 3. EDA
- Kontrak Bulanan: churn 54,6% (1.541 pelanggan), Tahunan 16,8% (923), Dua Tahun 9,7% (536).
- Tenure: churn 54,6% pada 0–12 bulan, turun ke 41,1% (13–24), 28,4% (25–48), dan 6,7% (49–72).
- Kombinasi Bulanan + keluhan ≥ 3: 171 pelanggan dengan churn 71,9%.
- Region bukan pembeda utama: churn hanya berkisar 31,4% (Bali-Nusa Tenggara) – 37,3% (Kalimantan).
- Halaman EDA: `streamlit run sesi2/eda_churn.py` dan `streamlit run sesi2/app_eda.py`.

### 4. Visualization
| Chart | Alasan pemilihan |
|---|---|
| Donut komposisi risiko | Komposisi 3 kategori yang menjumlah 100%; angka total di tengah |
| Bar horizontal risiko tinggi per region | Perbandingan 5 kategori, diurutkan agar peringkat langsung terbaca |
| Line churn aktual vs prediksi per kuartal | Tren berurutan waktu; dua seri untuk memeriksa kecocokan model |
| Stacked bar risiko per kontrak | Komposisi di dalam setiap kelompok kontrak |
| Histogram probabilitas + garis threshold | Distribusi skor model dan batas keputusan 0,5 |
| Scatter region aktual vs prediksi | Hubungan dua ukuran; ukuran titik = jumlah pelanggan |
| Gauge (tab simulasi) | KPI tunggal satu pelanggan, berwarna sesuai kategori risiko |

### 5. Final ML Model
- `models/churn_model.joblib` — scikit-learn Pipeline (imputasi median, standarisasi, one-hot
  encoding) + GradientBoostingClassifier, versi 1.0.0, scikit-learn 1.9.1.
- Kinerja holdout (dari `model_metadata.json`): ROC-AUC 0,789 · akurasi 0,718 · F1 0,579.
- Model disediakan fasilitator; **tidak** dilatih ulang, di-tuning, atau dibandingkan.

### 6. Model Loading & Inference
- `utils/model_utils.py`: `load_model()` (joblib), `load_metadata()`, `validate_input()`
  (kolom wajib, kategori dikenal, tipe numerik), `prepare_input()` (urutan fitur), `predict()`.
- Dashboard memuat model sekali saat start dan menjalankan batch inference untuk 3.000 pelanggan.
- Data mentah langsung dikirim ke Pipeline — tidak ada scaling/encoding manual agar
  preprocessing tidak terjadi dua kali.

### 7. Prediction Output
- `churn_probability` (0–1), `predicted_churn` (threshold 0,5), dan `risk_category`:
  Rendah < 0,30 · Sedang 0,30–0,60 · Tinggi ≥ 0,60.
- Catatan: kelas prediksi dan kategori risiko adalah dua cara membaca probabilitas yang sama —
  pelanggan dengan probabilitas 0,55 sudah "Prediksi churn" tetapi masih kategori Sedang.

### 8. Interactive Components
- Sidebar: 5 filter (region, segmen, jenis kontrak, kategori risiko prediksi, rentang tenure).
- Tab Simulasi: 9 input pelanggan + tombol "Jalankan prediksi".
- Chart Plotly: hover, zoom, legend toggle; tabel prioritas dengan sort, filter, dan paging.

### 9. Callback / App Logic
- `update_overview`: 5 Input filter → 9 Output (4 KPI, 6 chart, data tabel, judul tabel).
  Data sudah berisi prediksi, jadi callback hanya memotong DataFrame (respons cepat).
- `run_prediction`: Input tombol + 9 State → teks hasil + gauge. Memakai State agar model hanya
  dipanggil saat tombol diklik; `PreventUpdate` jika ada input angka kosong.
- Filter kosong → "Tidak ada data — longgarkan filter" tanpa error.

### 10. Dashboard
- 4 KPI: Pelanggan · Churn rate aktual · Rata-rata prob. churn · Risiko tinggi (+ tagihan berisiko).
- 6 chart Plotly + gauge; 1 tabel prioritas retensi (200 probabilitas tertinggi).
- Alur baca: header (pertanyaan bisnis) → KPI → chart → tabel; simulasi di tab terpisah.
- Memenuhi semua Dashboard Minimum Requirements (Tabel 8.2): ≥ 4 KPI, ≥ 3 visualisasi,
  ≥ 2 filter, ≥ 1 tampilan prediksi, ≥ 1 tabel detail.

### 11. Insight & Recommendation
Lihat Bagian 2.

---

## Bagian 2 — Insight dan rekomendasi (§8.7)

Format: **temuan + angka** → implikasi/rekomendasi. Angka dari dashboard tanpa filter.

1. **624 pelanggan (20,8%) berada pada kategori risiko tinggi**, dengan total tagihan
   **Rp288,3 juta per bulan**.
   → Ini adalah ukuran pendapatan yang dipertaruhkan; jadikan dasar anggaran program retensi.
2. **96% pelanggan risiko tinggi berkontrak Bulanan.**
   → Program migrasi ke kontrak Tahunan dengan insentif diskon untuk pelanggan Bulanan berisiko tinggi
   (churn kontrak Tahunan 16,8% vs Bulanan 54,6%).
3. **Rata-rata tenure pelanggan risiko tinggi 14,0 bulan**, hampir separuh rata-rata seluruh
   pelanggan (26,5 bulan).
   → Perkuat onboarding dan kontak proaktif pada 12 bulan pertama.
4. **Rata-rata keluhan pelanggan risiko tinggi 1,78**, dibanding 1,13 untuk seluruh pelanggan.
   → SLA penyelesaian keluhan maksimal 48 jam untuk pelanggan berisiko tinggi.
5. **Proporsi risiko tinggi pada Transfer Bank 27,4%**, tertinggi dibanding E-Wallet 19,6%,
   Kartu Kredit 17,0%, dan Virtual Account 16,6%.
   → Dorong autodebit atau e-wallet untuk mengurangi friksi pembayaran.
6. **Sumatera memiliki proporsi risiko tinggi terbesar (24,3%)**, Bali-Nusa Tenggara terendah (18,3%).
   → Prioritaskan kampanye retensi regional di Sumatera.
7. **Kategori risiko konsisten dengan data historis:** churn aktual 78,5% pada kelompok Tinggi,
   45,4% pada Sedang, dan 9,7% pada Rendah.
   → Kategori risiko layak dipakai untuk memprioritaskan tindakan. *Catatan:* sebagian data ini
   juga dipakai saat model dilatih, jadi angka ini cenderung lebih baik daripada kinerja pada
   pelanggan baru; acuan yang adil adalah metrik holdout (ROC-AUC 0,789).
8. **134 pelanggan risiko tinggi masih aktif (belum churn).**
   → Jadikan 134 pelanggan ini daftar kerja pertama tim retensi minggu ini (tabel prioritas di
   dashboard, filter Risiko = Tinggi).

### Keterbatasan
- Data sintetis untuk latihan; pola nyata bisa berbeda.
- Model tidak dilatih ulang; kinerja holdout ROC-AUC 0,789 dan F1 0,579 berarti masih ada
  pelanggan yang salah klasifikasi — perlakukan prediksi sebagai "berisiko", bukan kepastian.
- Korelasi bukan sebab-akibat: kontrak Bulanan berhubungan dengan churn, tetapi belum terbukti
  bahwa memindahkan pelanggan ke kontrak Tahunan *menyebabkan* churn turun — uji dengan pilot.
- Threshold 0,5 adalah pilihan default; menurunkannya menandai lebih banyak pelanggan
  (pada 200 pelanggan baru: 98 pada 0,3 vs 25 pada 0,7 — lihat `sesi4/latihan_sesi4.py`).

---

## Bagian 3 — Struktur presentasi 5–7 menit (§8.8)

Sebelum presentasi: ubah `debug=True` menjadi `debug=False` di baris terakhir
`sesi8/app_final_dashboard.py`, jalankan aplikasinya, dan buka di browser.

| Waktu | Bagian | Yang dikatakan / ditunjukkan |
|---|---|---|
| 0:00–0:30 | Masalah bisnis | Churn 34,9% — satu dari tiga pelanggan berhenti. Pertanyaan: siapa yang berisiko dan apa tindakannya? (tunjuk header dashboard) |
| 0:30–1:15 | Data & model | 3.000 pelanggan, 2023–2025. Model final Gradient Boosting (ROC-AUC 0,789), dimuat — tidak dilatih ulang. Probabilitas → kategori Rendah/Sedang/Tinggi (0,30 dan 0,60). |
| 1:15–3:15 | Tur dashboard | KPI (624 risiko tinggi, Rp288,3 jt/bln) → donut komposisi risiko → bar region → komposisi per kontrak → histogram + threshold → tabel prioritas. |
| 3:15–4:15 | Demo interaksi | (1) Filter Risiko: centang hanya **Tinggi** → semua chart dan tabel ikut berubah. (2) Tab **Simulasi Prediksi**, nilai default → klik *Jalankan prediksi* → 71,1%, Churn, Tinggi. Ubah kontrak ke **Dua Tahun** dan tenure **48** → turun ke Rendah. |
| 4:15–5:45 | Insight | Pilih 3–5 dari Bagian 2; urutan yang disarankan: no. 1 (skala masalah), 2 (kontrak), 3 (tenure), 4 (keluhan), 8 (daftar kerja). |
| 5:45–6:45 | Rekomendasi & keterbatasan | Migrasi kontrak, onboarding 12 bulan, SLA keluhan 48 jam, daftar 134 pelanggan. Batasan: data sintetis, model tidak sempurna, korelasi ≠ sebab-akibat. |

Pertanyaan yang mungkin muncul:
- *Kenapa 0,55 sudah "churn" tapi masih "Sedang"?* — kelas prediksi memakai threshold 0,5;
  kategori risiko memakai batas 0,30/0,60. Keduanya membaca probabilitas yang sama.
- *Seberapa bisa dipercaya modelnya?* — ROC-AUC holdout 0,789; cukup untuk memprioritaskan,
  bukan untuk memastikan.
- *Kenapa tidak melatih model sendiri?* — lingkup pelatihan: memuat model final, inference,
  dan menyajikan output.

---

## Bagian 4 — Checklist final submission (§8.10)

| Item | Status | Lokasi |
|---|---|---|
| Source code Streamlit/Dash | ✅ | `sesi1/` … `sesi8/` |
| Dataset / analytical data | ✅ | `data/customer_churn.csv`, `data/new_customers.csv` |
| EDA workflow | ✅ | `sesi2/eda_churn.py`, `sesi2/app_eda.py` |
| Interactive Plotly visualizations | ✅ | `sesi6/`, `sesi8/app_final_dashboard.py` |
| Final trained ML model | ✅ | `models/churn_model.joblib`, `models/model_metadata.json` |
| Model loading & inference workflow | ✅ | `utils/model_utils.py`, `sesi4/` |
| Prediction result/output | ✅ | Dashboard final: KPI, filter risiko, histogram, tabel, gauge |
| Interactive dashboard | ✅ | `sesi8/app_final_dashboard.py` |
| Filter/input components | ✅ | 5 filter sidebar + 9 input simulasi |
| Callback implementation | ✅ | `sesi7/`, `sesi8/` |
| Business insight | ✅ (draf) | Bagian 2 dokumen ini |
| Recommendation | ✅ (draf) | Bagian 2 dokumen ini |
| Presentasi final project | ⬜ | Latih dengan Bagian 3 |
