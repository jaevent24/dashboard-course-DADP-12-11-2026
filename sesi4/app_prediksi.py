# sesi4/app_prediksi.py — Aplikasi prediksi churn dengan model yang sudah dilatih
# Jalankan dari folder dashboard_course:  streamlit run sesi4/app_prediksi.py
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Prediksi Churn", page_icon="🔮", layout="wide")

WARNA_RISIKO = {"Rendah": "#1baf7a", "Sedang": "#eda100", "Tinggi": "#e04b3c"}


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "churn_model.joblib")


@st.cache_data
def load_meta():
    return json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))


@st.cache_data
def load_data_latih():
    df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
    df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())
    return df


def rupiah(x):
    return f"Rp{x:,.0f}".replace(",", ".")


def kategori_risiko(p, bins):
    for nama, (lo, hi) in bins.items():
        if lo <= p < hi or (hi == 1.0 and p == 1.0):
            return nama
    return "-"


def prediksi(df_input):
    hasil = df_input.copy()
    hasil["proba_churn"] = model.predict_proba(df_input[FITUR])[:, 1]
    hasil["prediksi"] = (hasil["proba_churn"] >= meta["decision_threshold"]).map(
        {True: "Churn", False: "Bertahan"})
    hasil["risiko"] = hasil["proba_churn"].apply(lambda p: kategori_risiko(p, meta["risk_bins"]))
    return hasil


@st.cache_data
def baca_csv(isi):
    import io
    return pd.read_csv(io.BytesIO(isi))


def siapkan_eda(data):
    """Siapkan data untuk tab EDA. Kolom churn dipakai jika ada; jika tidak, pakai prediksi model."""
    d = data.copy()
    if "signup_date" in d.columns:
        d["signup_date"] = pd.to_datetime(d["signup_date"], errors="coerce")
    if "total_charges" in d.columns:
        d["total_charges"] = d["total_charges"].fillna(d["total_charges"].median())
    if "churn" in d.columns:
        return d, False
    d["churn"] = (model.predict_proba(d[FITUR])[:, 1] >= meta["decision_threshold"]).astype(int)
    return d, True


try:
    model, meta = load_model(), load_meta()
except FileNotFoundError:
    st.error("Model tidak ditemukan. Pastikan file models/churn_model.joblib dan "
             "models/model_metadata.json ada di dalam folder dashboard_course.")
    st.stop()

FITUR = meta["feature_order"]
KAT = meta["categories"]
RNG = meta["numeric_ranges"]

# ---------- SIDEBAR: sumber data ----------
st.sidebar.header("📂 Data pelanggan")
file = st.sidebar.file_uploader("Unggah CSV pelanggan", type="csv",
                                help="Dipakai di tab EDA dan Banyak pelanggan. Wajib berisi kolom: "
                                     + ", ".join(FITUR) + ". Kolom churn dan signup_date opsional.")
upload = None
if file is not None:
    upload = baca_csv(file.getvalue())
    kurang_up = [c for c in FITUR if c not in upload.columns]
    if kurang_up:
        st.sidebar.error(f"Kolom tidak ada: {', '.join(kurang_up)}")
        upload = None
    else:
        st.sidebar.success(f"{file.name}: {len(upload):,} baris")
else:
    st.sidebar.caption("Tab EDA dan Banyak pelanggan akan terisi setelah CSV diunggah.")

# ---------- SIDEBAR: info model ----------
st.sidebar.divider()
st.sidebar.header("🤖 Tentang model")
st.sidebar.write(f"**{meta['model_name']}** v{meta['model_version']}")
st.sidebar.caption(meta["algorithm"])
m = meta["holdout_metrics"]
s1, s2, s3 = st.sidebar.columns(3)
s1.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
s2.metric("Akurasi", f"{m['accuracy']:.0%}")
s3.metric("F1", f"{m['f1']:.2f}")
st.sidebar.caption("Kategori risiko berdasarkan probabilitas churn:")
for nama, (lo, hi) in meta["risk_bins"].items():
    st.sidebar.write(f"- **{nama}**: {lo:.0%} – {hi:.0%}")

# ---------- HEADER ----------
st.title("🔮 Prediksi Customer Churn")
st.caption("Jelajahi data pelanggan, lalu prediksi churn untuk satu pelanggan atau banyak "
           "pelanggan sekaligus.")

tab0, tab1, tab2 = st.tabs(["📊 EDA", "👤 Satu pelanggan", "📂 Banyak pelanggan (CSV)"])

# ---------- TAB 0: EDA (dashboard Sesi 3 + latihan) ----------
with tab0:
    df, pakai_prediksi = None, False
    if upload is not None:
        df, pakai_prediksi = siapkan_eda(upload)
        sumber_eda = file.name
    else:
        st.info("Unggah CSV pelanggan di sidebar kiri untuk menampilkan EDA.")

    if df is not None:
        ada_tgl = "signup_date" in df.columns and df["signup_date"].notna().any()
        label_churn = "Churn rate (prediksi)" if pakai_prediksi else "Churn rate"
        st.caption(f"Sumber data: {sumber_eda}")
        if pakai_prediksi:
            st.info("File ini tidak punya kolom **churn**, jadi semua angka churn di tab ini "
                    "memakai **prediksi model** (probabilitas ≥ "
                    f"{meta['decision_threshold']:.0%} dianggap churn), bukan data aktual.")
        if "reset" not in st.session_state:
            st.session_state.reset = 0
        r = st.session_state.reset

        opsi_region = sorted(df["region"].unique())
        opsi_segment = sorted(df["segment"].unique())
        opsi_layanan = sorted(df["internet_service"].unique())
        tmin, tmax = int(df.tenure_months.min()), int(df.tenure_months.max())

        with st.expander("🎛️ Filter data", expanded=True):
            f1, f2, f3 = st.columns(3)
            region = f1.multiselect("Region", opsi_region, default=opsi_region,
                                    key=f"eda_region_{r}")
            segment = f2.multiselect("Segmen", opsi_segment, default=opsi_segment,
                                     key=f"eda_segment_{r}")
            layanan = f3.multiselect("Layanan internet", opsi_layanan, default=opsi_layanan,
                                     key=f"eda_layanan_{r}")
            f4, f5, f6 = st.columns(3)
            kontrak = f4.radio("Jenis kontrak", ["Semua"] + sorted(df["contract_type"].unique()),
                               horizontal=True, key=f"eda_kontrak_{r}")
            tenure = f5.slider("Tenure (bulan)", tmin, tmax, (tmin, tmax), key=f"eda_tenure_{r}")
            tgl = None
            if ada_tgl:
                tgl = f6.date_input("Periode pendaftaran",
                                    (df.signup_date.min().date(), df.signup_date.max().date()),
                                    key=f"eda_tgl_{r}")
            if st.button("🔄 Reset filter", key="eda_reset"):
                st.session_state.reset += 1
                st.rerun()

        mask = (df["region"].isin(region) & df["segment"].isin(segment)
                & df["internet_service"].isin(layanan)
                & df["tenure_months"].between(*tenure))
        if kontrak != "Semua":
            mask &= df["contract_type"] == kontrak
        if ada_tgl and isinstance(tgl, tuple) and len(tgl) == 2:
            mask &= df["signup_date"].between(pd.Timestamp(tgl[0]), pd.Timestamp(tgl[1]))
        dff = df[mask]
        st.caption(f"{len(dff):,} dari {len(df):,} pelanggan terpilih")

        if dff.empty:
            st.warning("Tidak ada data untuk kombinasi filter ini. Longgarkan filter di atas.")
        else:
            # KPI
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total pelanggan", f"{len(dff):,}")
            k2.metric(label_churn, f"{dff.churn.mean():.1%}",
                      delta=f"{(dff.churn.mean() - df.churn.mean()) * 100:+.1f} pp vs total",
                      delta_color="inverse")
            k3.metric("Rata-rata tagihan", rupiah(dff.monthly_charges.mean()))
            k4.metric("Rata-rata tenure", f"{dff.tenure_months.mean():.1f} bln")
            keluhan3 = (dff["num_complaints"] >= 3).mean()
            keluhan3_total = (df["num_complaints"] >= 3).mean()
            k5.metric("Keluhan ≥ 3", f"{keluhan3:.1%}",
                      delta=f"{(keluhan3 - keluhan3_total) * 100:+.1f} pp vs total",
                      delta_color="inverse", help="Persentase pelanggan dengan 3 keluhan atau lebih")

            st.write("")
            t_ring, t_seg, t_det = st.tabs(["📈 Ringkasan", "🔍 Segmentasi", "📋 Data Detail"])

            with t_ring:
                c1, c2 = st.columns([3, 2])
                with c1:
                    with st.container(border=True):
                        if ada_tgl:
                            tren = (dff.set_index("signup_date").resample("ME").size()
                                      .rename("pelanggan").reset_index())
                            fig_tren = px.line(tren, x="signup_date", y="pelanggan", markers=True,
                                               title="Pelanggan baru per bulan",
                                               color_discrete_sequence=["#2a78d6"])
                        else:
                            fig_tren = px.histogram(dff, x="tenure_months", nbins=24,
                                                    title="Sebaran tenure pelanggan",
                                                    color_discrete_sequence=["#2a78d6"])
                            fig_tren.update_layout(xaxis_title="Tenure (bulan)",
                                                   yaxis_title="Pelanggan")
                        st.plotly_chart(fig_tren.update_layout(height=320, template="plotly_white"),
                                        width="stretch")
                with c2:
                    with st.container(border=True):
                        fig_donut = px.pie(dff, names="segment", hole=0.55, title="Komposisi segmen",
                                           color_discrete_sequence=["#2a78d6", "#1baf7a", "#eda100"])
                        fig_donut.update_traces(textinfo="percent+label")
                        st.plotly_chart(fig_donut.update_layout(height=320, showlegend=False),
                                        width="stretch")
                c3, c4 = st.columns(2)
                with c3:
                    with st.container(border=True):
                        ck = (dff.groupby("contract_type", as_index=False)["churn"].mean()
                                 .sort_values("churn", ascending=False))
                        fig_bar = px.bar(ck, x="contract_type", y="churn", text_auto=".1%",
                                         title="Churn rate per jenis kontrak",
                                         color_discrete_sequence=["#eb6834"])
                        fig_bar.update_yaxes(tickformat=".0%", title=label_churn)
                        st.plotly_chart(fig_bar.update_layout(height=320, template="plotly_white",
                                                              xaxis_title=""), width="stretch")
                with c4:
                    with st.container(border=True):
                        if ada_tgl:
                            kq = (dff.groupby(dff["signup_date"].dt.to_period("Q").astype(str))["churn"]
                                     .mean().rename_axis("kuartal").reset_index())
                            fig_q = px.line(kq, x="kuartal", y="churn", markers=True,
                                            title="Churn rate per kuartal pendaftaran",
                                            color_discrete_sequence=["#eb6834"])
                        else:
                            kq = (dff.groupby("num_complaints")["churn"].mean()
                                     .rename_axis("keluhan").reset_index())
                            fig_q = px.line(kq, x="keluhan", y="churn", markers=True,
                                            title="Churn rate per jumlah keluhan",
                                            color_discrete_sequence=["#eb6834"])
                        fig_q.update_yaxes(tickformat=".0%", title=label_churn, rangemode="tozero")
                        st.plotly_chart(fig_q.update_layout(height=320, template="plotly_white",
                                                            xaxis_title=""), width="stretch")

            with t_seg:
                pilih_seg = st.selectbox("Segmentasi berdasarkan",
                                         ["region", "segment", "payment_method", "internet_service"],
                                         key="eda_seg")
                seg = (dff.groupby(pilih_seg).agg(pelanggan=("churn", "size"),
                                                  churn_rate=("churn", "mean"),
                                                  tagihan=("monthly_charges", "mean"))
                          .sort_values("churn_rate", ascending=False))
                a, b = st.columns(2)
                fig_seg = px.bar(seg.reset_index(), x=pilih_seg, y="churn_rate", text_auto=".1%",
                                 color_discrete_sequence=["#2a78d6"],
                                 title=f"Churn rate per {pilih_seg}")
                fig_seg.update_yaxes(tickformat=".0%")
                a.plotly_chart(fig_seg.update_layout(height=320, template="plotly_white"),
                               width="stretch")
                b.dataframe(seg.style.format({"churn_rate": "{:.1%}", "tagihan": rupiah})
                               .background_gradient(subset=["churn_rate"], cmap="Oranges"),
                            width="stretch")

            with t_det:
                cols = st.multiselect("Kolom", list(dff.columns), default=list(dff.columns)[:8],
                                      key="eda_cols")
                st.dataframe(dff[cols], width="stretch", height=360)
                st.download_button("⬇️ Unduh CSV", dff.to_csv(index=False).encode("utf-8"),
                                   file_name="churn_filtered.csv", mime="text/csv",
                                   key="eda_download")

# ---------- TAB 1: prediksi satu pelanggan ----------
with tab1:
    kiri, kanan = st.columns([3, 2], gap="large")
    with kiri:
        with st.form("form_pelanggan"):
            st.subheader("Data pelanggan")
            a, b = st.columns(2)
            region = a.selectbox("Region", KAT["region"])
            segment = b.selectbox("Segmen", KAT["segment"])
            contract = a.selectbox("Jenis kontrak", KAT["contract_type"])
            payment = b.selectbox("Metode pembayaran", KAT["payment_method"])
            internet = a.selectbox("Layanan internet", KAT["internet_service"])

            tenure = b.slider("Tenure (bulan)", int(RNG["tenure_months"][0]),
                              int(RNG["tenure_months"][1]), 12)
            monthly = a.number_input("Tagihan bulanan (Rp)", int(RNG["monthly_charges"][0]),
                                     int(RNG["monthly_charges"][1]), 500_000, step=10_000)
            total_auto = b.checkbox("Total tagihan = tagihan bulanan × tenure", value=True)
            total = b.number_input("Total tagihan (Rp)", 0, int(RNG["total_charges"][1]),
                                   500_000 * 12, step=100_000, disabled=total_auto)
            complaints = a.slider("Jumlah keluhan", int(RNG["num_complaints"][0]),
                                  int(RNG["num_complaints"][1]), 1)
            calls = b.slider("Panggilan ke customer support", int(RNG["support_calls"][0]),
                             int(RNG["support_calls"][1]), 2)
            kirim = st.form_submit_button("Prediksi churn", type="primary", width="stretch")

    with kanan:
        if not kirim:
            st.info("Isi data pelanggan di sebelah kiri, lalu klik **Prediksi churn**.")
        else:
            if total_auto:
                total = monthly * tenure
            satu = pd.DataFrame([dict(
                tenure_months=tenure, monthly_charges=monthly, total_charges=total,
                num_complaints=complaints, support_calls=calls, region=region,
                segment=segment, contract_type=contract, payment_method=payment,
                internet_service=internet)])
            h = prediksi(satu).iloc[0]
            p, risiko = h["proba_churn"], h["risiko"]

            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=p * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "Probabilitas churn"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": WARNA_RISIKO.get(risiko, "#2a78d6")},
                       "steps": [{"range": [lo * 100, hi * 100], "color": "#f2f2f2"}
                                 for lo, hi in meta["risk_bins"].values()],
                       "threshold": {"line": {"color": "black", "width": 2},
                                     "value": meta["decision_threshold"] * 100}}))
            st.plotly_chart(gauge.update_layout(height=280, margin=dict(t=60, b=10)),
                            width="stretch")

            pesan = f"Risiko **{risiko}** · diprediksi **{h['prediksi']}**"
            {"Tinggi": st.error, "Sedang": st.warning}.get(risiko, st.success)(pesan)
            st.caption(f"Total tagihan yang dipakai: {rupiah(total)}. "
                       f"Garis hitam = ambang keputusan {meta['decision_threshold']:.0%}.")

            # Bandingkan dengan churn rate historis pelanggan sejenis
            try:
                latih = load_data_latih()
                mirip = latih[(latih["contract_type"] == contract) & (latih["segment"] == segment)]
                if len(mirip):
                    st.caption(f"Sebagai pembanding: churn rate historis pelanggan "
                               f"{segment} dengan kontrak {contract} adalah "
                               f"{mirip['churn'].mean():.1%} ({len(mirip):,} pelanggan).")
            except FileNotFoundError:
                pass

# ---------- TAB 2: prediksi banyak pelanggan ----------
with tab2:
    if upload is None:
        st.info("Unggah CSV pelanggan di sidebar kiri untuk memprediksi banyak pelanggan sekaligus.")
        st.stop()
    data, sumber = upload, file.name

    kurang = [c for c in FITUR if c not in data.columns]
    if kurang:
        st.error(f"Kolom berikut tidak ada di {sumber}: {', '.join(kurang)}")
        st.stop()

    hasil = prediksi(data)
    st.caption(f"Sumber data: {sumber} · unggah file lain lewat sidebar")

    urutan = list(meta["risk_bins"].keys())
    jumlah = hasil["risiko"].value_counts().reindex(urutan, fill_value=0)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pelanggan", f"{len(hasil):,}")
    k2.metric("Diprediksi churn", f"{(hasil['prediksi'] == 'Churn').mean():.1%}")
    k3.metric("Risiko tinggi", f"{jumlah.get('Tinggi', 0):,}")
    k4.metric("Rata-rata probabilitas", f"{hasil['proba_churn'].mean():.1%}")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            fig_bar = px.bar(jumlah.rename_axis("risiko").reset_index(name="pelanggan"),
                             x="risiko", y="pelanggan", color="risiko", text_auto=True,
                             color_discrete_map=WARNA_RISIKO, title="Jumlah pelanggan per risiko")
            st.plotly_chart(fig_bar.update_layout(height=320, template="plotly_white",
                                                  showlegend=False, xaxis_title=""),
                            width="stretch")
    with c2:
        with st.container(border=True):
            fig_hist = px.histogram(hasil, x="proba_churn", nbins=20, color="risiko",
                                    color_discrete_map=WARNA_RISIKO,
                                    category_orders={"risiko": urutan},
                                    title="Sebaran probabilitas churn")
            fig_hist.update_xaxes(tickformat=".0%", title="Probabilitas churn")
            st.plotly_chart(fig_hist.update_layout(height=320, template="plotly_white",
                                                   yaxis_title="Pelanggan"), width="stretch")

    pilih = st.multiselect("Tampilkan risiko", urutan, default=urutan)
    tampil = hasil[hasil["risiko"].isin(pilih)].sort_values("proba_churn", ascending=False)
    st.dataframe(tampil.style.format({"proba_churn": "{:.1%}"}), width="stretch", height=360)
    st.download_button("⬇️ Unduh hasil prediksi (CSV)",
                       tampil.to_csv(index=False).encode("utf-8"),
                       file_name="prediksi_churn.csv", mime="text/csv")