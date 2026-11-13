# sesi2/eda_churn.py — EDA lengkap dengan Pandas + Matplotlib, ditampilkan sebagai halaman web
# Jalankan: streamlit run sesi2/eda_churn.py
from pathlib import Path
import matplotlib
matplotlib.use("Agg")            # render tanpa jendela; Streamlit yang menampilkan gambarnya
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="EDA Churn — Matplotlib", page_icon="📊", layout="wide")

# ---------- Gaya visual konsisten untuk semua grafik ----------
BIRU, ORANYE, HIJAU, KUNING, UNGU = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
TEKS, TEKS2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
STATUS = {"Bertahan": BIRU, "Churn": ORANYE}
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#b9b8b2", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.labelcolor": TEKS2, "xtick.color": TEKS2, "ytick.color": TEKS2,
})
KONTRAK = ["Bulanan", "Tahunan", "Dua Tahun"]


# ---------- Penempatan grafik di halaman (menggantikan simpan ke folder output) ----------
def _grid():
    """Menghasilkan slot kolom kiri/kanan bergantian: 2 grafik per baris."""
    while True:
        kiri, kanan = st.columns(2)
        yield kiri
        yield kanan


_slot = None


def bagian(judul, keterangan=""):
    """Mulai bagian baru (judul + grid 2 kolom baru)."""
    global _slot
    st.subheader(judul)
    if keterangan:
        st.caption(keterangan)
    _slot = _grid()


def simpan(fig, nama, lebar_penuh=False):
    """Tampilkan figure di halaman lalu tutup agar memori tidak menumpuk."""
    fig.tight_layout()
    wadah = st.container() if lebar_penuh else next(_slot)
    with wadah:
        st.pyplot(fig)
        st.caption(nama)
    plt.close(fig)


# =============== 1. LOAD DATA ===============
df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})

st.title("📊 EDA Customer Churn — Pandas + Matplotlib")
st.caption("Sesi 2 · Praktikum A: profiling, statistik, filtering, grouping, dan 16 visualisasi EDA")

# =============== 2. PROFILING ===============
st.header("1. Profiling data")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Baris", f"{df.shape[0]:,}")
m2.metric("Kolom", df.shape[1])
m3.metric("Duplikat customer_id", int(df["customer_id"].duplicated().sum()))
m4.metric("Missing total_charges", int(df["total_charges"].isna().sum()))
profil = pd.DataFrame({"tipe": df.dtypes.astype(str), "missing": df.isna().sum(),
                       "unik": df.nunique()})

# =============== 3. STATISTIK DESKRIPTIF ===============
num_cols = ["tenure_months", "monthly_charges", "total_charges", "num_complaints", "support_calls"]
c1, c2 = st.columns([2, 3])
c1.markdown("**Profil kolom**")
c1.dataframe(profil)
c2.markdown("**Statistik deskriptif variabel numerik**")
c2.dataframe(df[num_cols].describe().T.round(1))
c2.metric("Churn rate keseluruhan", "{:.1%}".format(df["churn"].mean()))

# =============== 4. CLEANING RINGAN ===============
df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())
df["kelompok_tenure"] = pd.cut(df["tenure_months"], bins=[0, 12, 24, 48, 72],
                               labels=["0-12", "13-24", "25-48", "49-72"])

# =============== 5. FILTERING ===============
st.header("2. Filtering & grouping")
berisiko = df[(df["contract_type"] == "Bulanan") & (df["num_complaints"] >= 3)]
f1, f2 = st.columns(2)
f1.metric("Bulanan & keluhan ≥ 3", f"{len(berisiko)} pelanggan",
          f"churn {berisiko['churn'].mean():.1%}", delta_color="off")
f2.metric("Pelanggan Fiber di Jawa",
          len(df.query("region == 'Jawa' and internet_service == 'Fiber'")))

# =============== 6. GROUPING ===============
churn_kontrak = (df.groupby("contract_type")
                   .agg(pelanggan=("customer_id", "count"), churn_rate=("churn", "mean"),
                        rata2_tagihan=("monthly_charges", "mean"))
                   .reindex(KONTRAK))
pivot = pd.pivot_table(df, values="churn", index="region", columns="segment", aggfunc="mean")
g1, g2 = st.columns(2)
g1.markdown("**Ringkasan per jenis kontrak**")
g1.dataframe(churn_kontrak.round(3))
g2.markdown("**Pivot churn rate region × segmen**")
g2.dataframe(pivot.round(3))

# =============== 7. VISUALISASI ===============
st.header("3. Galeri visualisasi EDA")

# 7.1 KPI cards — ringkasan angka utama
bagian("Ringkasan", "Berapa angka utama saat ini?")
kpi = [("Total pelanggan", f"{len(df):,}".replace(",", "."), "3 tahun data"),
       ("Churn rate", f"{df.churn.mean():.1%}".replace(".", ","), f"{df.churn.sum():,} pelanggan".replace(",", ".")),
       ("Rata-rata tagihan", f"Rp{df.monthly_charges.mean()/1000:,.0f} rb", "per bulan"),
       ("Rata-rata tenure", f"{df.tenure_months.mean():.1f} bln".replace(".", ","), "lama berlangganan")]
fig, axes = plt.subplots(1, 4, figsize=(12, 2.1))
for ax, (judul, nilai, sub), warna in zip(axes, kpi, [BIRU, ORANYE, HIJAU, UNGU]):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.02,rounding_size=0.06",
                                transform=ax.transAxes, facecolor="#f7f7f5", edgecolor="#dcdbd5"))
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.025, 0.9, boxstyle="square,pad=0",
                                transform=ax.transAxes, facecolor=warna, edgecolor="none"))
    ax.text(0.1, 0.72, judul, fontsize=10, color=TEKS2, transform=ax.transAxes)
    ax.text(0.1, 0.38, nilai, fontsize=20, fontweight="bold", color=TEKS, transform=ax.transAxes)
    ax.text(0.1, 0.15, sub, fontsize=9, color=TEKS2, transform=ax.transAxes)
simpan(fig, "01_kpi_cards", lebar_penuh=True)

# 7.2 Pie chart — proporsi status churn
bagian("Komposisi", "Bagaimana keseluruhan terbagi menjadi bagian?")
cnt = df["status"].value_counts().reindex(["Bertahan", "Churn"])
fig, ax = plt.subplots(figsize=(5, 4))
wedges, labels, persen = ax.pie(
    cnt, labels=[f"{k}\n{v:,} pelanggan".replace(",", ".") for k, v in cnt.items()],
    colors=[STATUS[k] for k in cnt.index], autopct="%1.1f%%", startangle=90,
    counterclock=False, pctdistance=0.68, labeldistance=1.12,
    wedgeprops=dict(edgecolor="white", linewidth=2), textprops=dict(color=TEKS))
for t in persen:                       # angka persen di dalam irisan: putih & tebal
    t.set_color("white"); t.set_fontweight("bold"); t.set_fontsize(12)
ax.set_title("Satu dari tiga pelanggan churn")
simpan(fig, "02_pie_status_churn")

# 7.3 Donut chart — komposisi pelanggan per jenis kontrak
kc = df["contract_type"].value_counts().reindex(KONTRAK)
fig, ax = plt.subplots(figsize=(5, 4))
ax.pie(kc, colors=[BIRU, HIJAU, KUNING], startangle=90, counterclock=False,
       wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
ax.text(0, 0.06, f"{len(df):,}".replace(",", "."), ha="center", fontsize=18, fontweight="bold")
ax.text(0, -0.16, "pelanggan", ha="center", color=TEKS2)
ax.legend([f"{k} {v/len(df):.0%}" for k, v in kc.items()], loc="center left",
          bbox_to_anchor=(0.95, 0.5), frameon=False)
ax.set_title("Separuh pelanggan berkontrak bulanan")
simpan(fig, "03_donut_kontrak")

# 7.4 Bar chart vertikal + garis rata-rata
bagian("Perbandingan", "Kelompok mana yang lebih tinggi/rendah?")
fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(churn_kontrak.index, churn_kontrak["churn_rate"] * 100, color=ORANYE, width=0.55)
ax.bar_label(bars, fmt="%.1f%%", padding=3, color=TEKS)
ax.axhline(df.churn.mean() * 100, color=TEKS2, linestyle="--", linewidth=1)
ax.text(2.33, df.churn.mean() * 100 + 1, "rata-rata 34,9%", ha="right", color=TEKS2, fontsize=9)
ax.set(ylabel="Churn rate (%)", ylim=(0, 65))
ax.set_title("Churn kontrak bulanan 3x lebih tinggi dari tahunan")
simpan(fig, "04_bar_churn_kontrak")

# 7.5 Bar chart horizontal terurut — region
reg = df.groupby("region")["churn"].mean().sort_values() * 100
fig, ax = plt.subplots(figsize=(6.5, 3.6))
bars = ax.barh(reg.index, reg.values, color=BIRU, height=0.55)
ax.bar_label(bars, fmt="%.1f%%", padding=3, color=TEKS)
ax.set(xlabel="Churn rate (%)", xlim=(0, 45))
ax.grid(axis="y", visible=False)
ax.set_title("Perbedaan churn antar-region relatif kecil")
simpan(fig, "05_barh_churn_region")

# 7.6 Grouped bar — segmen x kontrak
gb = df.pivot_table(values="churn", index="contract_type", columns="segment",
                    aggfunc="mean").reindex(KONTRAK) * 100
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(gb.index)); w = 0.26
for i, (seg, warna) in enumerate(zip(gb.columns, [BIRU, ORANYE, HIJAU])):
    ax.bar(x + (i - 1) * w, gb[seg], width=w - 0.02, color=warna, label=seg)
ax.set_xticks(x, gb.index); ax.set_ylabel("Churn rate (%)")
ax.legend(title="Segmen", frameon=False)
ax.set_title("Pola kontrak berlaku di semua segmen")
simpan(fig, "06_grouped_bar_segmen_kontrak")

# 7.7 Stacked bar 100% — komposisi status per metode pembayaran
sb = pd.crosstab(df["payment_method"], df["status"], normalize="index").sort_values("Churn") * 100
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh(sb.index, sb["Bertahan"], color=BIRU, label="Bertahan", height=0.55)
ax.barh(sb.index, sb["Churn"], left=sb["Bertahan"], color=ORANYE, label="Churn", height=0.55)
for i, v in enumerate(sb["Churn"]):
    ax.text(100 - v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontweight="bold")
ax.set(xlabel="Proporsi pelanggan (%)", xlim=(0, 100)); ax.grid(axis="y", visible=False)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
ax.set_title("Transfer Bank memiliki porsi churn terbesar")
simpan(fig, "07_stacked_bar_pembayaran")

# 7.8 Histogram overlay — distribusi tenure per status
bagian("Distribusi", "Bagaimana sebaran nilai satu variabel?")
fig, ax = plt.subplots(figsize=(7, 4))
for s in ["Bertahan", "Churn"]:
    ax.hist(df.loc[df.status == s, "tenure_months"], bins=24, alpha=0.65, color=STATUS[s],
            label=s, edgecolor="white")
ax.set(xlabel="Tenure (bulan)", ylabel="Jumlah pelanggan"); ax.legend(frameon=False)
ax.set_title("Churn menumpuk pada tenure pendek")
simpan(fig, "08_hist_tenure_status")

# 7.9 Box plot — tagihan per status
fig, ax = plt.subplots(figsize=(6, 4))
data_box = [df.loc[df.status == s, "monthly_charges"] / 1000 for s in ["Bertahan", "Churn"]]
bp = ax.boxplot(data_box, tick_labels=["Bertahan", "Churn"], patch_artist=True, widths=0.5,
                medianprops=dict(color=TEKS))
for patch, s in zip(bp["boxes"], ["Bertahan", "Churn"]):
    patch.set_facecolor(STATUS[s]); patch.set_alpha(0.55)
ax.set(ylabel="Tagihan bulanan (ribu Rp)")
ax.set_title("Pelanggan churn cenderung bertagihan sedikit lebih tinggi")
simpan(fig, "09_box_tagihan_status")

# 7.10 Violin plot — sebaran tenure per kontrak
fig, ax = plt.subplots(figsize=(6.5, 4))
vp = ax.violinplot([df.loc[df.contract_type == k, "tenure_months"] for k in KONTRAK],
                   showmedians=True)
for body, warna in zip(vp["bodies"], [BIRU, HIJAU, KUNING]):
    body.set_facecolor(warna); body.set_alpha(0.6)
for key in ("cmedians", "cmins", "cmaxes", "cbars"):
    vp[key].set_color(TEKS2)
ax.set_xticks([1, 2, 3], KONTRAK); ax.set_ylabel("Tenure (bulan)")
ax.set_title("Kontrak panjang identik dengan tenure panjang")
simpan(fig, "10_violin_tenure_kontrak")

# 7.11 Line chart — pelanggan baru per bulan + rata-rata bergerak 3 bulan
bagian("Tren", "Bagaimana nilai berubah dari waktu/urutan?")
bulanan = df.set_index("signup_date").resample("ME")["customer_id"].count()
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(bulanan.index, bulanan.values, color="#9dbbe6", linewidth=1.5, marker="o", markersize=3,
        label="Per bulan")
ax.plot(bulanan.index, bulanan.rolling(3).mean(), color=BIRU, linewidth=2.2,
        label="Rata-rata bergerak 3 bulan")
ax.set(ylabel="Pelanggan baru", ylim=(0, None)); ax.legend(frameon=False, loc="lower right")
ax.set_title("Akuisisi pelanggan stabil sekitar 80 per bulan")
simpan(fig, "11_line_signup")

# 7.12 Line chart — churn rate per kelompok tenure
kt = df.groupby("kelompok_tenure", observed=True)["churn"].mean() * 100
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(kt.index.astype(str), kt.values, color=ORANYE, marker="o", markersize=8, linewidth=2)
for xi, yi in zip(kt.index.astype(str), kt.values):
    ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center")
ax.set(xlabel="Kelompok tenure (bulan)", ylabel="Churn rate (%)", ylim=(0, 65))
ax.set_title("Churn turun tajam setelah 2 tahun berlangganan")
simpan(fig, "12_line_churn_tenure")

# 7.13 Stacked area — pelanggan baru per kuartal per segmen
area = (df.assign(kuartal=df.signup_date.dt.to_period("Q").dt.to_timestamp())
          .pivot_table(index="kuartal", columns="segment", values="customer_id", aggfunc="count"))
area = area[["Retail", "UMKM", "Korporat"]]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.stackplot(area.index, area.T.values, labels=area.columns, colors=[BIRU, ORANYE, HIJAU],
             alpha=0.85, edgecolor="white", linewidth=1)
ax.set(ylabel="Pelanggan baru per kuartal"); ax.legend(loc="upper left", ncol=3, frameon=False)
ax.set_title("Retail mendominasi akuisisi di setiap kuartal")
simpan(fig, "13_area_segmen")

# 7.14 Scatter — tenure vs tagihan, warna = status
bagian("Hubungan", "Apakah dua variabel saling berkaitan?")
fig, ax = plt.subplots(figsize=(7, 4.2))
for s in ["Bertahan", "Churn"]:
    d = df[df.status == s]
    ax.scatter(d["tenure_months"], d["monthly_charges"] / 1000, s=9, alpha=0.5,
               color=STATUS[s], label=s)
ax.set(xlabel="Tenure (bulan)", ylabel="Tagihan bulanan (ribu Rp)"); ax.legend(frameon=False)
ax.set_title("Titik churn mengumpul di sisi kiri (tenure pendek)")
simpan(fig, "14_scatter_tenure_tagihan")

# 7.15 Heatmap — churn rate region x segmen
fig, ax = plt.subplots(figsize=(6.5, 4))
im = ax.imshow(pivot.values * 100, cmap="Oranges", aspect="auto", vmin=20, vmax=55)
ax.set_xticks(range(len(pivot.columns)), pivot.columns)
ax.set_yticks(range(len(pivot.index)), pivot.index); ax.grid(False)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j] * 100
        ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v > 45 else TEKS)
fig.colorbar(im, ax=ax, label="Churn rate (%)")
ax.set_title("Korporat Sulawesi: churn tertinggi (52,6%)")
simpan(fig, "15_heatmap_region_segmen")

# 7.16 Heatmap korelasi — hubungan antar-variabel numerik
corr = df[num_cols + ["churn"]].corr()
fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
lbl = ["tenure", "tagihan", "total", "keluhan", "panggilan CS", "churn"]
ax.set_xticks(range(len(lbl)), lbl, rotation=35, ha="right"); ax.set_yticks(range(len(lbl)), lbl)
ax.grid(False)
for i in range(len(lbl)):
    for j in range(len(lbl)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(corr.values[i, j]) > 0.6 else TEKS, fontsize=9)
fig.colorbar(im, ax=ax, label="Korelasi Pearson")
ax.set_title("Keluhan berkorelasi positif, tenure negatif dengan churn")
simpan(fig, "16_heatmap_korelasi")
