# sesi7/app_dash_callback_latihan.py — Latihan mandiri Sesi 7 (port 8055)
# 1. dcc.Checklist internet_service -> ikut di filter_df()
# 2. Output kelima di callback utama: histogram tenure yang ikut terfilter
# 3. Klik titik scatter -> detail pelanggan di html.Div (custom_data + clickData)
# 4. dcc.Store menyimpan DataFrame hasil filter sekali; callback tabel membaca dari store
# 5. Tantangan: box/lasso select (selectedData) pada scatter -> filter tabel
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback, ctx, no_update
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})
WARNA = {"Bertahan": "#4C72B0", "Churn": "#C44E52"}
KARTU = {"flex": 1, "padding": "14px", "borderRadius": "10px", "background": "#F4F6FA",
         "textAlign": "center"}
TABEL_COLS = ["customer_id", "region", "segment", "contract_type", "internet_service",
              "tenure_months", "monthly_charges", "num_complaints", "status"]
LAYANAN = sorted(df.internet_service.unique())

app = Dash(__name__, title="Dash Callback — Latihan")
app.layout = html.Div(style={"maxWidth": "1200px", "margin": "auto", "fontFamily": "Arial"}, children=[
    html.H1("Dash Callback System — Latihan Sesi 7"),

    # ---- Panel filter ----
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "16px"},
             children=[
        html.Div([html.Label("Region"),
                  dcc.Dropdown(id="dd-region", options=sorted(df.region.unique()), multi=True,
                               placeholder="Semua region")]),
        html.Div([html.Label("Segmen (opsi bergantung region)"),
                  dcc.Dropdown(id="dd-segment", multi=True, placeholder="Semua segmen")]),
        html.Div([html.Label("Jenis kontrak"),
                  dcc.RadioItems(id="rd-kontrak", inline=True, value="Semua",
                                 options=["Semua", "Bulanan", "Tahunan", "Dua Tahun"])]),
        # Latihan 1: checklist layanan internet
        html.Div([html.Label("Layanan internet"),
                  dcc.Checklist(id="ck-layanan", options=LAYANAN, value=LAYANAN)]),
    ]),
    html.Div(style={"marginTop": "12px"}, children=[
        html.Label("Tenure (bulan)"),
        dcc.RangeSlider(id="rs-tenure", min=1, max=72, step=1, value=[1, 72],
                        marks={i: str(i) for i in (1, 12, 24, 36, 48, 60, 72)}),
    ]),
    html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center"}, children=[
        html.Label("Minimal jumlah keluhan:"),
        dcc.Input(id="in-keluhan", type="number", min=0, max=10, value=0, style={"width": "70px"}),
        html.Button("Terapkan", id="btn-terapkan", n_clicks=0),
        html.Button("Reset filter", id="btn-reset", n_clicks=0),
        html.Span(id="info-keluhan", style={"color": "#666"}),
    ]),
    dcc.Store(id="store-keluhan", data=0),
    dcc.Store(id="store-data"),                       # Latihan 4: hasil filter disimpan sekali

    # ---- KPI ----
    html.Div(id="kpi", style={"display": "flex", "gap": "12px", "margin": "16px 0"}),

    # ---- Charts ----
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
        dcc.Graph(id="g-bar"), dcc.Graph(id="g-scatter"),
        dcc.Graph(id="g-hist"),                       # Latihan 2
        html.Div(style={**KARTU, "textAlign": "left"}, children=[
            html.H4("Detail pelanggan (klik titik scatter)", style={"marginTop": 0}),
            html.Div(id="detail", children="Belum ada titik yang diklik."),   # Latihan 3
        ]),
    ]),
    html.P("Tip: klik batang region, atau pilih titik di scatter dengan Box/Lasso Select "
           "(ikon di pojok kanan atas chart), untuk memfilter tabel.", style={"color": "#666"}),
    html.H3(id="judul-tabel"),
    dash_table.DataTable(id="tbl", page_size=10, sort_action="native",
                         columns=[{"name": c, "id": c} for c in TABEL_COLS],
                         style_cell={"padding": "6px", "textAlign": "left"},
                         style_header={"fontWeight": "bold", "background": "#E9EDF5"},
                         style_data_conditional=[{"if": {"filter_query": "{status} = 'Churn'"},
                                                  "backgroundColor": "#FBE9EA"}]),
])


def filter_df(region, segment, kontrak, layanan, tenure, min_keluhan):
    d = df
    if region:
        d = d[d.region.isin(region)]
    if segment:
        d = d[d.segment.isin(segment)]
    if kontrak and kontrak != "Semua":
        d = d[d.contract_type == kontrak]
    d = d[d.internet_service.isin(layanan or [])]    # Latihan 1: kosong = tidak ada data
    d = d[d.tenure_months.between(tenure[0], tenure[1])]
    return d[d.num_complaints >= (min_keluhan or 0)]


# 1) Chained callback: opsi segmen bergantung pada region terpilih
@callback(Output("dd-segment", "options"), Input("dd-region", "value"))
def update_segment_options(region):
    d = df[df.region.isin(region)] if region else df
    return sorted(d.segment.unique())


# 2) Terapkan / reset (reset juga mengembalikan checklist layanan)
@callback(Output("store-keluhan", "data"), Output("in-keluhan", "value"),
          Output("dd-region", "value"), Output("dd-segment", "value"),
          Output("rd-kontrak", "value"), Output("ck-layanan", "value"),
          Output("rs-tenure", "value"),
          Input("btn-terapkan", "n_clicks"), Input("btn-reset", "n_clicks"),
          State("in-keluhan", "value"), prevent_initial_call=True)
def terapkan_atau_reset(n_apply, n_reset, nilai):
    if ctx.triggered_id == "btn-reset":
        return 0, 0, None, None, "Semua", LAYANAN, [1, 72]
    if nilai is None:
        raise PreventUpdate
    return nilai, *[no_update] * 6


# 3) Callback utama: 6 input -> KPI, 3 chart, info, dan store hasil filter
@callback(Output("kpi", "children"), Output("g-bar", "figure"), Output("g-scatter", "figure"),
          Output("g-hist", "figure"), Output("info-keluhan", "children"),
          Output("store-data", "data"),
          Input("dd-region", "value"), Input("dd-segment", "value"), Input("rd-kontrak", "value"),
          Input("ck-layanan", "value"), Input("rs-tenure", "value"), Input("store-keluhan", "data"))
def update_dashboard(region, segment, kontrak, layanan, tenure, min_keluhan):
    d = filter_df(region, segment, kontrak, layanan, tenure, min_keluhan)
    info = f"Filter keluhan aktif: ≥ {min_keluhan}"
    records = d[TABEL_COLS].to_dict("records")       # Latihan 4: disimpan ke dcc.Store
    if d.empty:
        kosong = px.scatter(title="Tidak ada data untuk filter ini")
        return [html.Div("Data kosong", style=KARTU)], kosong, kosong, kosong, info, records

    def kartu(j, v):
        return html.Div([html.Small(j), html.H2(v, style={"margin": "4px 0"})], style=KARTU)

    kpi = [kartu("Pelanggan", f"{len(d):,}"), kartu("Churn rate", f"{d.churn.mean():.1%}"),
           kartu("Rata-rata tagihan", f"Rp{d.monthly_charges.mean():,.0f}".replace(",", ".")),
           kartu("Rata-rata keluhan", f"{d.num_complaints.mean():.2f}")]

    reg = d.groupby("region", as_index=False)["churn"].mean().sort_values("churn", ascending=False)
    bar = px.bar(reg, x="region", y="churn", template="plotly_white", title="Churn rate per region",
                 labels={"churn": "Churn rate", "region": ""}, color_discrete_sequence=["#C44E52"])
    bar.update_yaxes(tickformat=".0%")

    # Latihan 3 & 5: custom_data membawa customer_id ke clickData / selectedData
    sc = px.scatter(d, x="tenure_months", y="monthly_charges", color="status", opacity=0.6,
                    color_discrete_map=WARNA, template="plotly_white", hover_name="customer_id",
                    custom_data=["customer_id"], title="Tenure vs tagihan",
                    labels={"tenure_months": "Tenure (bulan)", "monthly_charges": "Tagihan (Rp)"})
    sc.update_layout(dragmode="select")              # langsung siap box select

    # Latihan 2: histogram tenure ikut terfilter
    hist = px.histogram(d, x="tenure_months", color="status", nbins=24, barmode="overlay",
                        opacity=0.7, color_discrete_map=WARNA, template="plotly_white",
                        title="Distribusi tenure (terfilter)",
                        labels={"tenure_months": "Tenure (bulan)", "status": "Status"})
    return kpi, bar, sc, hist, info, records


# Latihan 3: klik titik scatter -> detail pelanggan
@callback(Output("detail", "children"), Input("g-scatter", "clickData"),
          prevent_initial_call=True)
def tampil_detail(click):
    if not click:
        raise PreventUpdate
    cid = click["points"][0]["customdata"][0]
    r = df.loc[df.customer_id == cid].iloc[0]
    return html.Ul([html.Li(f"ID: {r.customer_id}"), html.Li(f"Region: {r.region}"),
                    html.Li(f"Segmen: {r.segment}"), html.Li(f"Kontrak: {r.contract_type}"),
                    html.Li(f"Tenure: {r.tenure_months} bulan"),
                    html.Li(f"Keluhan: {r.num_complaints}"), html.Li(f"Status: {r.status}")])


# 4) Tabel membaca dari store; klik batang (region) atau box/lasso select (Latihan 5) memfilter
@callback(Output("tbl", "data"), Output("judul-tabel", "children"),
          Input("store-data", "data"), Input("g-bar", "clickData"),
          Input("g-scatter", "selectedData"))
def update_tabel(data, click, selected):
    d = pd.DataFrame(data or [], columns=TABEL_COLS)
    judul = "Detail pelanggan (semua region terpilih)"
    if ctx.triggered_id == "g-bar" and click:
        reg = click["points"][0]["x"]
        d = d[d.region == reg]
        judul = f"Detail pelanggan — region {reg}"
    elif ctx.triggered_id == "g-scatter" and selected and selected.get("points"):
        ids = [p["customdata"][0] for p in selected["points"]]
        d = d[d.customer_id.isin(ids)]
        judul = f"Detail pelanggan — {len(d)} titik terpilih di scatter"
    return d.to_dict("records"), judul


if __name__ == "__main__":
    app.run(debug=True, port=8055)
