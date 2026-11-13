# sesi4/latihan_sesi4.py — Latihan mandiri Sesi 4
# Jalankan: streamlit run sesi4/latihan_sesi4.py
# 1. Threshold keputusan sebagai slider (0,3–0,7) di sidebar -> dampak ke jumlah "Prediksi churn"
# 2. CSV berisi 3 pelanggan dengan contract_type salah ("Mingguan") -> pesan validasi muncul
# 3. Tabel ringkasan jumlah pelanggan per region × risk_category (pd.crosstab)
# 4. Tantangan: potensi pendapatan berisiko = jumlah monthly_charges pelanggan kategori Tinggi
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))                       # agar folder utils bisa di-import
from utils.model_utils import load_model, load_metadata, validate_input, predict  # noqa: E402

st.set_page_config(page_title="Latihan Sesi 4", page_icon="📝", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_meta():
    return load_metadata()


def rupiah(x):
    return f"Rp{x:,.0f}".replace(",", ".")


st.title("📝 Latihan Mandiri Sesi 4 — Inference model final")

# ---------- Latihan 1: threshold sebagai slider ----------
meta = dict(get_meta())                          # salinan: file metadata asli tidak berubah
threshold = st.sidebar.slider("Threshold keputusan", 0.3, 0.7, meta["decision_threshold"], 0.05,
                              help="Probabilitas ≥ threshold dianggap 'Prediksi churn'.")
meta["decision_threshold"] = threshold
st.sidebar.caption("Threshold hanya mengubah kelas prediksi (churn/bertahan). Kategori risiko "
                   f"tetap memakai risk bins: {get_meta()['risk_bins']}.")

# ---------- Sumber data (Latihan 2 memakai CSV salah) ----------
SUMBER = {"Contoh data/new_customers.csv (200 pelanggan)": ROOT / "data" / "new_customers.csv",
          "Latihan 2: sesi4/contoh_kontrak_salah.csv (3 pelanggan, 1 kontrak salah)":
              ROOT / "sesi4" / "contoh_kontrak_salah.csv"}
pilih = st.radio("Sumber data", list(SUMBER) + ["Upload CSV sendiri"])
if pilih == "Upload CSV sendiri":
    up = st.file_uploader("Upload CSV (format sama dengan data/new_customers.csv)", type="csv")
    if up is None:
        st.info("Pilih file CSV untuk diprediksi.")
        st.stop()
    data = pd.read_csv(up)
else:
    data = pd.read_csv(SUMBER[pilih])

# Latihan 2: validasi sebelum prediksi
errs = validate_input(data, meta)
if errs:
    st.error("Data tidak valid, prediksi dibatalkan:")
    for e in errs:
        st.error(e)
    st.dataframe(data, hide_index=True)
    st.stop()

hasil = predict(get_model(), data, meta)

# ---------- KPI (Latihan 1 + 4) ----------
tinggi = hasil[hasil.risk_category == "Tinggi"]
berisiko = tinggi["monthly_charges"].sum()                                    # Latihan 4
k1, k2, k3, k4 = st.columns(4)
k1.metric("Pelanggan diproses", f"{len(hasil):,}")
k2.metric(f"Prediksi churn (threshold {threshold:.2f})", f"{hasil.predicted_churn.sum():,}")
k3.metric("Risiko tinggi", f"{len(tinggi):,}")
k4.metric("Pendapatan bulanan berisiko", rupiah(berisiko),
          help="Jumlah monthly_charges pelanggan berkategori risiko Tinggi.")

# Latihan 1: dampak threshold — jumlah prediksi churn untuk setiap nilai slider
st.subheader("Latihan 1 — Dampak threshold terhadap jumlah prediksi churn")
proba = hasil["churn_probability"]
dampak = pd.DataFrame({"threshold": [round(t, 2) for t in
                                     [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]]})
dampak["prediksi_churn"] = [(proba >= t).sum() for t in dampak["threshold"]]
a, b = st.columns([2, 1])
a.bar_chart(dampak.set_index("threshold"), y="prediksi_churn")
b.dataframe(dampak, hide_index=True)
st.caption("Threshold lebih rendah → lebih banyak pelanggan ditandai churn (lebih banyak yang "
           "ditangani, tetapi lebih banyak salah sasaran). Threshold lebih tinggi → sebaliknya.")

# ---------- Latihan 3: crosstab region × risk_category ----------
st.subheader("Latihan 3 — Jumlah pelanggan per region × kategori risiko")
ct = pd.crosstab(hasil.region, hasil.risk_category, margins=True, margins_name="Total")
ct = ct.reindex(columns=[c for c in ["Rendah", "Sedang", "Tinggi", "Total"] if c in ct.columns])
st.dataframe(ct.style.background_gradient(subset=[c for c in ["Tinggi"] if c in ct.columns],
                                          cmap="Reds"))

# ---------- Latihan 4: rincian pendapatan berisiko ----------
st.subheader("Latihan 4 — Potensi pendapatan bulanan berisiko (kategori Tinggi)")
per_region = (tinggi.groupby("region")["monthly_charges"].agg(["count", "sum"])
                    .rename(columns={"count": "pelanggan", "sum": "tagihan_bulanan"})
                    .sort_values("tagihan_bulanan", ascending=False))
st.dataframe(per_region.style.format({"tagihan_bulanan": rupiah}))
st.markdown(f"**Total: {rupiah(berisiko)} per bulan** dari {len(tinggi)} pelanggan risiko tinggi.")
