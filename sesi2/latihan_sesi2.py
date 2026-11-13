# sesi2/latihan_sesi2.py — Latihan mandiri Sesi 2 no. 1–4, ditampilkan sebagai halaman web
# Jalankan: streamlit run sesi2/latihan_sesi2.py
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Latihan Sesi 2", page_icon="📝", layout="wide")

WARNA = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"]
STATUS = {"Bertahan": "#2a78d6", "Churn": "#eb6834"}
plt.rcParams.update({"axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False})


def simpan(fig, wadah):
    """Tampilkan figure di wadah Streamlit lalu tutup (menggantikan simpan ke folder output)."""
    fig.tight_layout()
    wadah.pyplot(fig)
    plt.close(fig)


df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})

st.title("📝 Latihan Mandiri Sesi 2")

# ---------- Latihan 1: pie layanan, donut region, lalu bar terurut ----------
st.subheader("Latihan 1 — Pie, donut, atau bar terurut?")
s = df["internet_service"].value_counts()
r = df["region"].value_counts()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].pie(s, labels=s.index, autopct="%1.1f%%", colors=WARNA[:3], startangle=90,
            wedgeprops=dict(edgecolor="white"))
axes[0].set_title("Pie: layanan internet (3 kategori)")
axes[1].pie(r, labels=r.index, autopct="%1.0f%%", colors=WARNA, startangle=90,
            wedgeprops=dict(width=0.4, edgecolor="white"))
axes[1].set_title("Donut: region (5 kategori)")
rs = r.sort_values()
bars = axes[2].barh(rs.index, rs.values, color=WARNA[0])
axes[2].bar_label(bars, padding=3)
axes[2].set_title("Bar terurut: region")
simpan(fig, st)
st.info("Untuk 5 region, bar horizontal terurut lebih mudah dibaca daripada donut karena "
        "perbedaan panjang batang langsung terlihat.")

kiri, kanan = st.columns(2)

# ---------- Latihan 2: stacked bar 100% status per layanan internet ----------
ct = pd.crosstab(df["internet_service"], df["status"], normalize="index") * 100
with kiri:
    st.subheader("Latihan 2 — Stacked bar 100% per layanan")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(ct.index, ct["Bertahan"], color=STATUS["Bertahan"], label="Bertahan", height=0.55)
    ax.barh(ct.index, ct["Churn"], left=ct["Bertahan"], color=STATUS["Churn"], label="Churn", height=0.55)
    for i, v in enumerate(ct["Churn"]):
        ax.text(100 - v / 2, i, f"{v:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.set(xlabel="Proporsi pelanggan (%)", xlim=(0, 100))
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.set_title("Layanan internet bukan pembeda churn")
    simpan(fig, st)
    st.markdown("**Churn per layanan (%)**")
    st.dataframe(ct["Churn"].round(1).rename("churn_%"))

# ---------- Latihan 3: box plot support_calls per status ----------
with kanan:
    st.subheader("Latihan 3 — Box plot panggilan CS")
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([df.loc[df.status == s, "support_calls"] for s in ["Bertahan", "Churn"]],
                    tick_labels=["Bertahan", "Churn"], patch_artist=True, widths=0.5)
    for patch, s in zip(bp["boxes"], ["Bertahan", "Churn"]):
        patch.set_facecolor(STATUS[s]); patch.set_alpha(0.55)
    ax.set(ylabel="Jumlah panggilan CS")
    ax.set_title("Panggilan CS per status churn")
    simpan(fig, st)
    st.markdown("**Median panggilan CS**")
    st.dataframe(df.groupby("status")["support_calls"].median().rename("median"))

# ---------- Latihan 4: churn rate per kuartal pendaftaran ----------
st.subheader("Latihan 4 — Churn rate per kuartal pendaftaran")
q = df.groupby(df.signup_date.dt.to_period("Q"))["churn"].mean() * 100
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(q.index.astype(str), q.values, marker="o", color="#eb6834", linewidth=2)
ax.set(xlabel="Kuartal pendaftaran", ylabel="Churn rate (%)", ylim=(0, None))
ax.tick_params(axis="x", rotation=45)
ax.set_title("Churn rate per kuartal pendaftaran")
c1, c2 = st.columns([3, 1])
simpan(fig, c1)
c2.markdown("**Churn per kuartal (%)**")
c2.dataframe(q.round(1).rename("churn_%").rename_axis("kuartal").reset_index().astype({"kuartal": str}),
             hide_index=True)
