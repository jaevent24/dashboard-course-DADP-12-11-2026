# sesi6/latihan_sesi6.py — Latihan mandiri Sesi 6 (jalankan: python sesi6/latihan_sesi6.py)
# 1. Bar horizontal churn rate per payment_method, terurut, + garis rata-rata (add_vline)
# 2. Bubble chart: scatter dengan size="support_calls"
# 3. Sunburst region -> segmen -> kontrak (values = jumlah pelanggan)
# 4. Gauge churn rate keseluruhan (go.Indicator mode="gauge+number")
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from plotly_galeri import ROOT, STATUS, KATEGORI, load  # pakai ulang load() & palet Sesi 6


def buat_latihan(df):
    fig = {}
    rata2 = df.churn.mean()

    # 1. Bar horizontal terurut + garis rata-rata nasional
    pm = (df.groupby("payment_method", as_index=False)["churn"].mean()
            .sort_values("churn"))                      # terkecil di bawah -> tertinggi di atas
    f = px.bar(pm, x="churn", y="payment_method", orientation="h", text_auto=".1%",
               color_discrete_sequence=["#eb6834"],
               labels={"churn": "Churn rate", "payment_method": ""},
               title="Transfer Bank memiliki churn rate tertinggi")
    f.add_vline(x=rata2, line_dash="dot",
                annotation_text=f"rata-rata {rata2:.1%}", annotation_position="top")
    f.update_xaxes(tickformat=".0%", range=[0, 0.45])
    fig["L1_bar_payment"] = f

    # 2. Bubble chart — ukuran titik = jumlah panggilan CS
    f = px.scatter(df, x="tenure_months", y="tagihan_ribu", color="status", size="support_calls",
                   size_max=18, opacity=0.5, color_discrete_map=STATUS, hover_name="customer_id",
                   hover_data={"support_calls": True, "num_complaints": True,
                               "tagihan_ribu": ":.0f"},
                   labels={"tenure_months": "Tenure (bulan)", "tagihan_ribu": "Tagihan (ribu Rp)",
                           "support_calls": "Panggilan CS"},
                   title="Bubble: ukuran = jumlah panggilan CS")
    fig["L2_bubble"] = f

    # 3. Sunburst 3 tingkat
    cnt = (df.groupby(["region", "segment", "contract_type"]).size()
             .reset_index(name="pelanggan"))
    f = px.sunburst(cnt, path=["region", "segment", "contract_type"], values="pelanggan",
                    color="region", color_discrete_sequence=KATEGORI,
                    title="Sunburst: region → segmen → kontrak (luas = jumlah pelanggan)")
    f.update_traces(insidetextorientation="radial")
    fig["L3_sunburst"] = f

    # 4. Gauge churn rate keseluruhan
    f = go.Figure(go.Indicator(mode="gauge+number", value=rata2 * 100,
                               number={"suffix": "%", "valueformat": ".1f"},
                               title={"text": "Churn rate keseluruhan"},
                               gauge={"axis": {"range": [0, 100]},
                                      "bar": {"color": "#eb6834"}}))
    fig["L4_gauge"] = f

    for f in fig.values():
        f.update_layout(template="plotly_white")
    return fig


if __name__ == "__main__":
    OUT = ROOT / "sesi6" / "output"
    OUT.mkdir(exist_ok=True)
    figs = buat_latihan(load())
    for nama, f in figs.items():
        f.write_html(OUT / f"{nama}.html", include_plotlyjs="cdn")
    print(f"{len(figs)} chart latihan tersimpan di", OUT)

# Cara membaca sunburst (latihan 3):
# - Lingkar dalam = region, lingkar tengah = segmen, lingkar luar = jenis kontrak.
# - Luas sudut tiap irisan sebanding dengan jumlah pelanggan; anak selalu berada di dalam
#   rentang sudut induknya, jadi irisan Retail di bawah Jawa adalah bagian dari Jawa.
# - Klik sebuah irisan untuk memperbesar cabang tersebut; klik tengah untuk kembali.
# - Contoh bacaan: Jawa adalah irisan region terbesar, dan di hampir setiap region
#   segmen Retail paling besar, dengan kontrak Bulanan mendominasi lingkar luar.
