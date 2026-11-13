# sesi5/app_dash_latihan.py — Latihan mandiri Sesi 5
# 1. KPI kelima: rata-rata keluhan
# 2. Grafik kedua (churn rate per region), berdampingan dengan CSS grid
# 3. DataTable dengan sorting dan paging
# 4. Gaya kartu dipindah ke assets/style.css dan dipakai lewat className
from pathlib import Path

import pandas as pd
from dash import Dash, html, dcc, dash_table

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])

ringkas = (df.groupby("contract_type")
             .agg(pelanggan=("customer_id", "count"), churn_rate=("churn", "mean"))
             .reindex(["Bulanan", "Tahunan", "Dua Tahun"])
             .reset_index())
ringkas["churn_rate"] = (ringkas["churn_rate"] * 100).round(1)

per_region = (df.groupby("region")["churn"].mean().mul(100).round(1)
                .sort_values(ascending=False).reset_index())

# assets_folder default = folder "assets" di sebelah file ini (sesi5/assets/)
app = Dash(__name__, title="Dash Latihan Sesi 5")


def kartu(judul, nilai):
    # Latihan 4: gaya dari style.css, bukan dict style
    return html.Div([html.P(judul), html.H2(nilai)], className="kartu")


def bar_figure(x, y, warna, judul_y):
    return {
        "data": [{"type": "bar", "x": x, "y": y, "marker": {"color": warna},
                  "text": [f"{v}%" for v in y], "textposition": "outside"}],
        "layout": {"yaxis": {"title": judul_y, "range": [0, max(y) * 1.2]},
                   "height": 350, "margin": {"t": 20}},
    }


app.layout = html.Div(style={"fontFamily": "Arial, sans-serif", "maxWidth": "1100px",
                             "margin": "auto", "padding": "20px"}, children=[
    html.H1("Dashboard Churn — Versi Dash (Latihan)"),
    html.P("Layout statis: 5 KPI cards, dua grafik berdampingan, dan tabel dengan sort & paging."),

    # Latihan 1: lima kartu KPI
    html.Div(style={"display": "flex", "gap": "12px"}, children=[
        kartu("Total pelanggan", f"{len(df):,}"),
        kartu("Churn rate", f"{df.churn.mean():.1%}"),
        kartu("Rata-rata tenure", f"{df.tenure_months.mean():.1f} bln"),
        kartu("Region", df.region.nunique()),
        kartu("Rata-rata keluhan", f"{df.num_complaints.mean():.2f}"),
    ]),

    # Latihan 2: dua grafik berdampingan dengan CSS grid
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
        html.Div([
            html.H3("Churn rate per jenis kontrak"),
            dcc.Graph(figure=bar_figure(ringkas["contract_type"], ringkas["churn_rate"],
                                        "#C44E52", "Churn rate (%)")),
        ]),
        html.Div([
            html.H3("Churn rate per region"),
            dcc.Graph(figure=bar_figure(per_region["region"], per_region["churn"],
                                        "#4C72B0", "Churn rate (%)")),
        ]),
    ]),

    # Latihan 3: sorting + paging
    html.H3("Ringkasan per kontrak"),
    dash_table.DataTable(data=ringkas.to_dict("records"),
                         columns=[{"name": c, "id": c} for c in ringkas.columns],
                         sort_action="native", page_size=5,
                         style_cell={"padding": "6px", "textAlign": "left"},
                         style_header={"fontWeight": "bold", "background": "#E9EDF5"}),

    html.Hr(),
    html.Small("Sumber: data latihan sintetis customer_churn.csv"),
])

if __name__ == "__main__":
    app.run(debug=True, port=8050)
