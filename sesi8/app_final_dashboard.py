# sesi8/app_final_dashboard.py — FINAL DASHBOARD PROJECT (Dash + Plotly + callback + ML inference)
#
# Navigasi lewat sidebar pop-up (tombol ☰ di kiri atas) dengan 4 halaman:
#   1. Data Overview          — profil dataset historis (ukuran, kamus data, missing value, statistik)
#   2. EDA                    — eksplorasi pola churn pada data historis (dengan filter)
#   3. Prediksi Single        — prediksi 1 pelanggan dari form input + analisis what-if
#   4. Prediksi Multi         — upload file CSV/Excel, hasil prediksi HANYA dari isi file tersebut
#
# Jalankan dari root project:  python sesi8/app_final_dashboard.py  →  http://127.0.0.1:8053
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback, ctx, no_update
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.model_utils import load_model, load_metadata, validate_input, predict  # noqa: E402

# ============ 1. MODEL REGISTRY ============
# CATATAN: saat ini baru ada SATU model (Gradient Boosting v1.0.0 di models/churn_model.joblib).
# Dropdown "Pilih model" di halaman Prediksi Single & Multi sudah disiapkan untuk banyak model.
# Untuk menambah model baru (mis. Logistic Regression / Random Forest):
#   1. Latih & simpan Pipeline-nya, mis. models/churn_model_rf.joblib, beserta metadatanya
#      models/model_metadata_rf.json (format sama dengan model_metadata.json).
#   2. Tambahkan entri baru ke MODEL_REGISTRY di bawah — dropdown otomatis ikut bertambah.
MODEL_REGISTRY = {
    "gb_v1": {"model_path": ROOT / "models" / "churn_model.joblib",
              "meta_path": ROOT / "models" / "model_metadata.json"},
    # "rf_v1": {"model_path": ROOT / "models" / "churn_model_rf.joblib",
    #           "meta_path": ROOT / "models" / "model_metadata_rf.json"},
}
MODELS = {}
for key, cfg in MODEL_REGISTRY.items():
    if cfg["model_path"].exists() and cfg["meta_path"].exists():
        meta = load_metadata(cfg["meta_path"])
        MODELS[key] = {"model": load_model(cfg["model_path"]), "meta": meta,
                       "label": f"{meta['model_name']} v{meta['model_version']}"}
if not MODELS:
    raise FileNotFoundError("Tidak ada model ditemukan. Jalankan dulu: python models/prepare_final_model.py")
DEFAULT_MODEL = next(iter(MODELS))
MODEL_OPTIONS = [{"label": m["label"], "value": k} for k, m in MODELS.items()]
META = MODELS[DEFAULT_MODEL]["meta"]                  # kontrak input (fitur & kategori) yang sama
CATS = META["categories"]

# ============ 2. DATA HISTORIS (dimuat sekali saat aplikasi start) ============
RAW = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])  # apa adanya
DF = RAW.copy()
DF["total_charges"] = DF["total_charges"].fillna(DF["monthly_charges"] * DF["tenure_months"])
DF["status"] = DF["churn"].map({0: "Bertahan", 1: "Churn"})
DF["periode"] = DF["signup_date"].dt.to_period("Q").astype(str)
# probabilitas historis per model → dipakai untuk menempatkan prediksi single dalam konteks populasi
HIST_PROBA = {k: predict(m["model"], DF, m["meta"])["churn_probability"].to_numpy()
              for k, m in MODELS.items()}

NUM_COLS = META["numeric_features"]
CAT_COLS = META["categorical_features"]
LABELS = {"tenure_months": "Tenure (bulan)", "monthly_charges": "Tagihan bulanan",
          "total_charges": "Total tagihan", "num_complaints": "Jumlah keluhan",
          "support_calls": "Panggilan CS", "region": "Region", "segment": "Segmen",
          "contract_type": "Jenis kontrak", "payment_method": "Metode pembayaran",
          "internet_service": "Layanan internet", "churn": "Churn", "customer_id": "ID pelanggan",
          "signup_date": "Tanggal daftar"}
DESKRIPSI = {"customer_id": "ID unik pelanggan", "signup_date": "Tanggal pelanggan mendaftar",
             "region": "Wilayah layanan", "segment": "Segmen bisnis pelanggan",
             "contract_type": "Durasi kontrak", "payment_method": "Cara pembayaran tagihan",
             "internet_service": "Jenis layanan internet", "tenure_months": "Lama berlangganan (bulan)",
             "monthly_charges": "Tagihan per bulan (Rp)", "total_charges": "Akumulasi tagihan (Rp)",
             "num_complaints": "Jumlah keluhan tercatat", "support_calls": "Jumlah panggilan ke CS",
             "churn": "TARGET — 1 = berhenti berlangganan"}

RISK_ORDER = ["Rendah", "Sedang", "Tinggi"]
RISK_COLOR = {"Rendah": "#55A868", "Sedang": "#DD8452", "Tinggi": "#C44E52"}
STATUS_COLOR = {"Bertahan": "#4C72B0", "Churn": "#C44E52"}
SARAN = {"Tinggi": "Prioritaskan retensi: hubungi dalam 48 jam, tawarkan kontrak tahunan.",
         "Sedang": "Pantau: kirim survei kepuasan dan penawaran loyalitas.",
         "Rendah": "Pertahankan layanan standar."}

# ============ 3. STYLE & KOMPONEN BANTU ============
C = {"bg": "#F5F7FB", "card": "white", "ink": "#1F2A44", "muted": "#6B7280", "accent": "#2F5DA8",
     "line": "#E5E7EB"}
CARD = {"background": C["card"], "borderRadius": "12px", "padding": "16px",
        "boxShadow": "0 1px 3px rgba(0,0,0,.08)"}
GRID2 = {"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))", "gap": "12px"}
ROW = {"display": "flex", "gap": "12px", "marginBottom": "12px", "flexWrap": "wrap"}
BTN = {"background": C["accent"], "color": "white", "border": 0, "padding": "10px 18px",
       "borderRadius": "8px", "cursor": "pointer", "fontWeight": 600}
BTN_GHOST = {**BTN, "background": "white", "color": C["accent"], "border": f"1px solid {C['accent']}"}
TABLE_STYLE = dict(style_cell={"padding": "6px", "fontSize": "13px", "textAlign": "left",
                               "fontFamily": "inherit"},
                   style_header={"fontWeight": "bold", "background": "#E9EDF5"},
                   style_table={"overflowX": "auto"})
RISK_TABLE_COND = [
    {"if": {"filter_query": "{risk_category} = 'Tinggi'", "column_id": "risk_category"},
     "color": "white", "backgroundColor": RISK_COLOR["Tinggi"]},
    {"if": {"filter_query": "{risk_category} = 'Sedang'", "column_id": "risk_category"},
     "backgroundColor": "#FBE3D3"}]


def label(t):
    return html.Label(t, style={"fontWeight": 600, "fontSize": "13px"})


def kpi(judul, nilai, sub=""):
    return html.Div(style={**CARD, "flex": "1 1 160px"}, children=[
        html.Div(judul, style={"color": C["muted"], "fontSize": "13px"}),
        html.Div(nilai, style={"fontSize": "26px", "fontWeight": 700, "color": C["ink"]}),
        html.Div(sub, style={"color": C["muted"], "fontSize": "12px"})])


def card(*children, **style):
    return html.Div(style={**CARD, **style}, children=list(children))


def section(judul, sub=None):
    return html.Div(style={"margin": "4px 0 12px"}, children=[
        html.H3(judul, style={"margin": 0}),
        html.Div(sub, style={"color": C["muted"], "fontSize": "14px"}) if sub else None])


def rupiah(x):
    return f"Rp{x:,.0f}".replace(",", ".")


def empty_fig(msg="Tidak ada data"):
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, font={"size": 14, "color": C["muted"]})
    f.update_layout(template="plotly_white", xaxis_visible=False, yaxis_visible=False, height=320)
    return f


def table(id_, columns, **kw):
    return dash_table.DataTable(id=id_, columns=columns, page_size=10, sort_action="native",
                                filter_action="native", **TABLE_STYLE, **kw)


def model_info(key):
    m = MODELS[key]["meta"]
    met = m.get("holdout_metrics", {})
    return html.Div(style={"fontSize": "13px", "color": C["muted"]}, children=[
        html.Div(m.get("algorithm", "")),
        html.Div(" · ".join(f"{k.upper()}: {v:.3f}" for k, v in met.items())),
        html.Div(f"Threshold churn: {m['decision_threshold']:.0%} · Risiko: " + ", ".join(
            f"{r} {lo:.0%}–{hi:.0%}" for r, (lo, hi) in m["risk_bins"].items()))])


def model_picker(id_):
    return card(label("Pilih model"),
                dcc.Dropdown(id=id_, options=MODEL_OPTIONS, value=DEFAULT_MODEL, clearable=False),
                html.Div(id=f"{id_}-info", style={"marginTop": "8px"}), marginBottom="12px")


# ============ 4. HALAMAN: DATA OVERVIEW ============
def build_overview():
    n_rows, n_cols = RAW.shape
    missing = RAW.isna().sum()
    kpis = html.Div(style=ROW, children=[
        kpi("Jumlah baris", f"{n_rows:,}", "pelanggan historis"),
        kpi("Jumlah kolom", f"{n_cols}", f"{len(NUM_COLS)} numerik · {len(CAT_COLS)} kategorikal"),
        kpi("Periode daftar", f"{RAW.signup_date.min():%b %Y} – {RAW.signup_date.max():%b %Y}"),
        kpi("Missing value", f"{int(missing.sum()):,}", f"{missing.sum() / RAW.size:.2%} dari seluruh sel"),
        kpi("Duplikat ID", f"{RAW.customer_id.duplicated().sum():,}"),
        kpi("Churn rate", f"{RAW.churn.mean():.1%}", f"{int(RAW.churn.sum()):,} pelanggan churn")])

    kamus = pd.DataFrame({
        "kolom": RAW.columns,
        "tipe": [("kategorikal" if c in CAT_COLS else "numerik" if c in NUM_COLS
                  else "target" if c == "churn" else "tanggal" if c == "signup_date" else "ID")
                 for c in RAW.columns],
        "deskripsi": [DESKRIPSI.get(c, "") for c in RAW.columns],
        "missing": missing.values,
        "unik": RAW.nunique().values,
        "contoh": [str(RAW[c].dropna().iloc[0]) for c in RAW.columns]})

    stats = RAW[NUM_COLS].describe().T.reset_index().rename(columns={"index": "fitur"})
    stats["fitur"] = stats["fitur"].map(LABELS)
    stats = stats.round(1)

    miss = missing[missing > 0].sort_values()
    g_miss = (px.bar(x=miss.values, y=miss.index, orientation="h", text_auto=True,
                     template="plotly_white", title="Missing value per kolom",
                     labels={"x": "Jumlah baris kosong", "y": ""},
                     color_discrete_sequence=[C["accent"]])
              if len(miss) else empty_fig("Tidak ada missing value"))
    g_miss.add_annotation(text="total_charges kosong diisi = tagihan bulanan × tenure",
                          xref="paper", yref="paper", x=1, y=-0.25, showarrow=False,
                          font={"size": 11, "color": C["muted"]})

    tgt = RAW.churn.map({0: "Bertahan", 1: "Churn"}).value_counts().reset_index()
    g_tgt = px.bar(tgt, x="churn", y="count", color="churn", text_auto=True,
                   color_discrete_map=STATUS_COLOR, template="plotly_white",
                   title="Distribusi target (churn)", labels={"churn": "", "count": "Pelanggan"})
    g_tgt.update_layout(showlegend=False)

    per_cat = pd.concat([RAW[c].value_counts().rename_axis("nilai").reset_index().assign(fitur=LABELS[c])
                         for c in CAT_COLS])
    g_cat = px.bar(per_cat, x="count", y="nilai", facet_col="fitur", facet_col_wrap=5,
                   orientation="h", template="plotly_white", height=360,
                   title="Komposisi fitur kategorikal", color_discrete_sequence=[C["accent"]],
                   labels={"count": "", "nilai": ""})
    g_cat.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    g_cat.update_yaxes(matches=None, showticklabels=True)

    preview = RAW.assign(signup_date=RAW.signup_date.dt.strftime("%Y-%m-%d"))
    return html.Div([
        section("Data Overview", "Profil dataset historis data/customer_churn.csv yang dipakai "
                                 "untuk melatih model"),
        kpis,
        card(html.H4("Kamus data", style={"marginTop": 0}),
             dash_table.DataTable(data=kamus.to_dict("records"),
                                  columns=[{"name": c.capitalize(), "id": c} for c in kamus.columns],
                                  **TABLE_STYLE), marginBottom="12px"),
        html.Div(style=GRID2, children=[
            card(dcc.Graph(figure=g_tgt)),
            card(dcc.Graph(figure=g_miss))]),
        card(dcc.Graph(figure=g_cat), marginTop="12px"),
        card(html.H4("Statistik deskriptif fitur numerik", style={"marginTop": 0}),
             dash_table.DataTable(data=stats.to_dict("records"),
                                  columns=[{"name": c, "id": c} for c in stats.columns], **TABLE_STYLE),
             marginTop="12px"),
        card(html.H4("Pratinjau data mentah", style={"marginTop": 0}),
             table("tbl-raw", [{"name": LABELS.get(c, c), "id": c} for c in preview.columns],
                   data=preview.to_dict("records")), marginTop="12px"),
    ])


# ============ 5. HALAMAN: EDA ============
eda_filters = card(
    html.Div(style={"display": "grid", "gap": "12px",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))"}, children=[
        html.Div([label("Region"), dcc.Dropdown(id="f-region", options=CATS["region"], multi=True,
                                                placeholder="Semua")]),
        html.Div([label("Segmen"), dcc.Dropdown(id="f-segment", options=CATS["segment"], multi=True,
                                                placeholder="Semua")]),
        html.Div([label("Jenis kontrak"), dcc.Dropdown(id="f-kontrak", options=CATS["contract_type"],
                                                       multi=True, placeholder="Semua")]),
        html.Div([label("Tenure (bulan)"),
                  dcc.RangeSlider(id="f-tenure", min=1, max=72, value=[1, 72], step=1,
                                  marks={1: "1", 24: "24", 48: "48", 72: "72"})])]),
    marginBottom="12px")

page_eda = html.Div([
    section("Exploratory Data Analysis", "Pola churn pada data historis — gunakan filter untuk "
                                         "memfokuskan analisis"),
    eda_filters,
    html.Div(id="eda-kpi", style=ROW),
    card(html.H4("Insight otomatis", style={"marginTop": 0}), html.Ul(id="eda-insight"),
         marginBottom="12px"),
    html.Div(style=GRID2, children=[
        card(label("Churn rate berdasarkan fitur kategorikal"),
             dcc.Dropdown(id="eda-cat", options=[{"label": LABELS[c], "value": c} for c in CAT_COLS],
                          value="contract_type", clearable=False),
             dcc.Graph(id="g-eda-cat")),
        card(label("Distribusi fitur numerik per status churn"),
             dcc.Dropdown(id="eda-num", options=[{"label": LABELS[c], "value": c} for c in NUM_COLS],
                          value="tenure_months", clearable=False),
             dcc.Graph(id="g-eda-num")),
        card(dcc.Graph(id="g-eda-trend")),
        card(dcc.Graph(id="g-eda-corr")),
        card(dcc.Graph(id="g-eda-complaint")),
        card(dcc.Graph(id="g-eda-scatter"))]),
    card(dcc.Graph(id="g-eda-heat"), marginTop="12px"),
])


# ============ 6. HALAMAN: PREDIKSI SINGLE ============
def num_input(id_, lab, val, mn, mx, step=1):
    return html.Div([label(lab), dcc.Input(id=id_, type="number", value=val, min=mn, max=mx,
                                           step=step, style={"width": "100%", "padding": "6px"})])


def dd_input(id_, lab, opts):
    return html.Div([label(lab), dcc.Dropdown(id=id_, options=opts, value=opts[0], clearable=False)])


page_single = html.Div([
    section("Prediksi Single Observasi", "Isi profil satu pelanggan untuk melihat probabilitas churn-nya"),
    model_picker("m-single"),
    html.Div(style=GRID2, children=[
        card(html.H4("Profil pelanggan", style={"marginTop": 0}),
             html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"}, children=[
                 dd_input("p-region", "Region", CATS["region"]),
                 dd_input("p-segment", "Segmen", CATS["segment"]),
                 dd_input("p-contract", "Kontrak", CATS["contract_type"]),
                 dd_input("p-payment", "Pembayaran", CATS["payment_method"]),
                 dd_input("p-service", "Layanan", CATS["internet_service"]),
                 num_input("p-tenure", "Tenure (bulan)", 6, 1, 72),
                 num_input("p-monthly", "Tagihan bulanan (Rp)", 450000, 100000, 2000000, 10000),
                 num_input("p-complaints", "Jumlah keluhan", 2, 0, 10),
                 num_input("p-calls", "Panggilan CS", 3, 0, 15)]),
             html.Small("Total tagihan dihitung otomatis = tagihan bulanan × tenure.",
                        style={"color": C["muted"]}),
             html.Br(), html.Br(),
             html.Button("Jalankan prediksi", id="btn-predict", n_clicks=0, style=BTN)),
        card(html.H4("Hasil prediksi", style={"marginTop": 0}),
             html.Div(id="pred-output", children=html.P("Isi form lalu klik tombol.",
                                                        style={"color": C["muted"]})),
             dcc.Graph(id="g-gauge", figure=empty_fig("Belum ada prediksi"), style={"height": "260px"})),
        card(dcc.Graph(id="g-whatif-tenure", figure=empty_fig("Analisis what-if muncul setelah prediksi"))),
        card(dcc.Graph(id="g-whatif-complaint", figure=empty_fig("Analisis what-if muncul setelah prediksi"))),
    ]),
])


# ============ 7. HALAMAN: PREDIKSI MULTI (UPLOAD FILE) ============
TEMPLATE_COLS = ["customer_id"] + META["feature_order"]
RESULT_COLS = [("ID", "customer_id"), ("Region", "region"), ("Segmen", "segment"),
               ("Kontrak", "contract_type"), ("Tenure", "tenure_months"),
               ("Tagihan/bln", "monthly_charges"), ("Keluhan", "num_complaints"),
               ("Panggilan CS", "support_calls"), ("Prob. churn", "churn_probability"),
               ("Prediksi", "prediksi"), ("Risiko", "risk_category")]

page_multi = html.Div([
    section("Prediksi Multi Observasi", "Upload file pelanggan — semua grafik & tabel di halaman ini "
                                        "HANYA menampilkan data dari file yang di-upload"),
    model_picker("m-multi"),
    card(dcc.Upload(id="upload", multiple=False, accept=".csv,.xlsx,.xls",
                    children=html.Div(["Tarik & lepas file di sini atau ",
                                       html.A("pilih file", style={"color": C["accent"],
                                                                   "fontWeight": 600}),
                                       html.Div("Format: CSV / Excel · kolom wajib: "
                                                + ", ".join(META["feature_order"]),
                                                style={"fontSize": "12px", "color": C["muted"],
                                                       "marginTop": "4px"})]),
                    style={"border": f"2px dashed {C['accent']}", "borderRadius": "10px",
                           "padding": "24px", "textAlign": "center", "cursor": "pointer",
                           "background": "#F8FAFF"}),
         html.Div(style={"display": "flex", "gap": "8px", "marginTop": "10px", "flexWrap": "wrap"},
                  children=[html.Button("Unduh template CSV", id="btn-template", style=BTN_GHOST),
                            html.Button("Unduh file contoh (200 pelanggan)", id="btn-sample",
                                        style=BTN_GHOST)]),
         html.Div(id="upload-status", style={"marginTop": "10px"}), marginBottom="12px"),
    html.Div(id="multi-results", style={"display": "none"}, children=[
        html.Div(id="multi-kpi", style=ROW),
        html.Div(style=GRID2, children=[
            card(dcc.Graph(id="g-multi-donut")),
            card(dcc.Graph(id="g-multi-hist")),
            card(dcc.Graph(id="g-multi-region")),
            card(dcc.Graph(id="g-multi-contract"))]),
        html.Div(id="multi-eval", style={"marginTop": "12px"}),
        card(html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                             "flexWrap": "wrap", "gap": "8px"}, children=[
                 html.H4("Hasil prediksi per pelanggan (urut probabilitas tertinggi)", style={"margin": 0}),
                 html.Button("Unduh hasil prediksi (CSV)", id="btn-download", style=BTN)]),
             html.Br(),
             table("tbl-multi",
                   [{"name": n, "id": i, **({"type": "numeric", "format": {"specifier": ".1%"}}
                                            if i == "churn_probability" else {})} for n, i in RESULT_COLS],
                   style_data_conditional=RISK_TABLE_COND),
             marginTop="12px"),
    ]),
    dcc.Store(id="upload-store"),
    dcc.Download(id="dl-template"), dcc.Download(id="dl-sample"), dcc.Download(id="dl-result"),
])


# ============ 8. LAYOUT: HEADER + SIDEBAR POP-UP ============
PAGES = [("overview", "📊  Data Overview"), ("eda", "🔍  EDA"),
         ("single", "👤  Prediksi Single Observasi"), ("multi", "📁  Prediksi Multi Observasi")]
DRAWER = {"position": "fixed", "top": 0, "left": 0, "height": "100vh", "width": "280px",
          "background": "white", "boxShadow": "2px 0 12px rgba(0,0,0,.15)", "zIndex": 1001,
          "padding": "20px 16px", "boxSizing": "border-box", "transition": "transform .25s ease",
          "display": "flex", "flexDirection": "column"}
BACKDROP = {"position": "fixed", "inset": 0, "background": "rgba(15,23,42,.35)", "zIndex": 1000}
NAV_BTN = {"display": "block", "width": "100%", "textAlign": "left", "padding": "12px 14px",
           "marginBottom": "6px", "border": 0, "borderRadius": "8px", "cursor": "pointer",
           "fontSize": "14px", "background": "transparent", "color": C["ink"]}
NAV_ACTIVE = {**NAV_BTN, "background": "#E8EEF9", "color": C["accent"], "fontWeight": 700}

drawer = html.Div(id="drawer", style={**DRAWER, "transform": "translateX(-100%)"}, children=[
    html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                    "marginBottom": "16px"}, children=[
        html.H3("Menu", style={"margin": 0}),
        html.Button("✕", id="btn-close", n_clicks=0,
                    style={"border": 0, "background": "none", "fontSize": "20px", "cursor": "pointer"})]),
    *[html.Button(t, id=f"nav-{k}", n_clicks=0, style=NAV_BTN) for k, t in PAGES],
    html.Div(style={"marginTop": "auto", "fontSize": "12px", "color": C["muted"]}, children=[
        html.Hr(), html.Div(f"{len(MODELS)} model tersedia"),
        *[html.Div(f"• {m['label']}") for m in MODELS.values()]]),
])

app = Dash(__name__, title="Final Dashboard — Customer Churn")
server = app.server                                   # untuk deployment (gunicorn)
app.layout = html.Div(style={"background": C["bg"], "minHeight": "100vh", "padding": "16px",
                             "fontFamily": "Inter, Arial, sans-serif", "color": C["ink"]}, children=[
    dcc.Store(id="page", data="overview"),
    html.Div(id="backdrop", n_clicks=0, style={**BACKDROP, "display": "none"}),
    drawer,
    html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px", "marginBottom": "16px"},
             children=[
        html.Button("☰", id="btn-menu", n_clicks=0, title="Buka menu",
                    style={**BTN, "fontSize": "20px", "padding": "6px 14px"}),
        html.Div([html.H2("Customer Churn Intelligence Dashboard", style={"margin": 0}),
                  html.Div(id="page-subtitle", style={"color": C["muted"]})])]),
    html.Div(id="page-overview", children=build_overview()),
    html.Div(id="page-eda", children=page_eda, style={"display": "none"}),
    html.Div(id="page-single", children=page_single, style={"display": "none"}),
    html.Div(id="page-multi", children=page_multi, style={"display": "none"}),
])


# ============ 9. CALLBACKS: NAVIGASI ============
@callback(Output("drawer", "style"), Output("backdrop", "style"), Output("page", "data"),
          Input("btn-menu", "n_clicks"), Input("btn-close", "n_clicks"), Input("backdrop", "n_clicks"),
          *[Input(f"nav-{k}", "n_clicks") for k, _ in PAGES], prevent_initial_call=True)
def toggle_drawer(*_):
    trig = ctx.triggered_id
    is_open = trig == "btn-menu"
    page = trig.removeprefix("nav-") if trig.startswith("nav-") else no_update
    return ({**DRAWER, "transform": "translateX(0)" if is_open else "translateX(-100%)"},
            {**BACKDROP, "display": "block" if is_open else "none"}, page)


@callback(*[Output(f"page-{k}", "style") for k, _ in PAGES],
          *[Output(f"nav-{k}", "style") for k, _ in PAGES], Output("page-subtitle", "children"),
          Input("page", "data"))
def show_page(page):
    shows = [{"display": "block" if k == page else "none"} for k, _ in PAGES]
    navs = [NAV_ACTIVE if k == page else NAV_BTN for k, _ in PAGES]
    return *shows, *navs, dict(PAGES)[page].split("  ", 1)[1]


@callback(Output("m-single-info", "children"), Input("m-single", "value"))
def single_model_info(key):
    return model_info(key)


@callback(Output("m-multi-info", "children"), Input("m-multi", "value"))
def multi_model_info(key):
    return model_info(key)


# ============ 10. CALLBACKS: EDA ============
def apply_filters(region, segment, kontrak, tenure):
    d = DF
    if region:
        d = d[d.region.isin(region)]
    if segment:
        d = d[d.segment.isin(segment)]
    if kontrak:
        d = d[d.contract_type.isin(kontrak)]
    return d[d.tenure_months.between(*tenure)]


@callback(Output("eda-kpi", "children"), Output("eda-insight", "children"),
          Output("g-eda-cat", "figure"), Output("g-eda-num", "figure"), Output("g-eda-trend", "figure"),
          Output("g-eda-corr", "figure"), Output("g-eda-complaint", "figure"),
          Output("g-eda-scatter", "figure"), Output("g-eda-heat", "figure"),
          Input("f-region", "value"), Input("f-segment", "value"), Input("f-kontrak", "value"),
          Input("f-tenure", "value"), Input("eda-cat", "value"), Input("eda-num", "value"))
def update_eda(region, segment, kontrak, tenure, cat, num):
    d = apply_filters(region, segment, kontrak, tenure)
    if d.empty:
        f = empty_fig("Tidak ada data — longgarkan filter")
        return [kpi("Pelanggan", "0")], [html.Li("Tidak ada data.")], f, f, f, f, f, f, f

    kpis = [kpi("Pelanggan", f"{len(d):,}", f"{len(d) / len(DF):.0%} dari total"),
            kpi("Churn rate", f"{d.churn.mean():.1%}", f"total data: {DF.churn.mean():.1%}"),
            kpi("Rata-rata tenure", f"{d.tenure_months.mean():.1f} bln"),
            kpi("Rata-rata tagihan", rupiah(d.monthly_charges.mean()), "per bulan"),
            kpi("Pendapatan hilang", rupiah(d.loc[d.churn == 1, "monthly_charges"].sum()),
                "per bulan dari pelanggan churn")]

    insights = []
    for c in CAT_COLS:
        r = d.groupby(c)["churn"].mean().sort_values()
        if len(r) > 1:
            insights.append(html.Li([html.B(LABELS[c]), f": churn tertinggi pada '{r.index[-1]}' "
                                     f"({r.iloc[-1]:.1%}) vs terendah '{r.index[0]}' ({r.iloc[0]:.1%})"]))
    corr = d[NUM_COLS + ["churn"]].corr()["churn"].drop("churn")
    top = corr.abs().idxmax()
    insights.append(html.Li([html.B("Fitur numerik paling berkorelasi"), f": {LABELS[top]} "
                             f"(r = {corr[top]:+.2f}, {'naik' if corr[top] > 0 else 'turun'} "
                             f"seiring kenaikan risiko churn)"]))

    rc = (d.groupby(cat).agg(rate=("churn", "mean"), n=("churn", "size")).reset_index()
            .sort_values("rate"))
    g_cat = px.bar(rc, x="rate", y=cat, orientation="h", text_auto=".1%", template="plotly_white",
                   hover_data={"n": True}, color="rate", color_continuous_scale="Reds",
                   labels={"rate": "Churn rate", cat: "", "n": "Pelanggan"})
    g_cat.add_vline(x=d.churn.mean(), line_dash="dash", annotation_text="rata-rata")
    g_cat.update_layout(coloraxis_showscale=False, margin={"t": 30})
    g_cat.update_xaxes(tickformat=".0%")

    g_num = px.histogram(d, x=num, color="status", barmode="overlay", opacity=0.65, nbins=30,
                         marginal="box", color_discrete_map=STATUS_COLOR, template="plotly_white",
                         labels={num: LABELS[num], "status": ""})
    g_num.update_layout(margin={"t": 30})

    tr = d.groupby("periode").agg(rate=("churn", "mean"), n=("churn", "size")).reset_index()
    g_trend = px.line(tr, x="periode", y="rate", markers=True, template="plotly_white",
                      hover_data={"n": True}, title="Churn rate per kuartal pendaftaran",
                      labels={"periode": "Kuartal pendaftaran", "rate": "Churn rate", "n": "Pelanggan"})
    g_trend.update_yaxes(tickformat=".0%")

    cm = d[NUM_COLS + ["churn"]].corr().round(2)
    cm.index = cm.columns = [LABELS[c] for c in cm.columns]
    g_corr = px.imshow(cm, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                       template="plotly_white", title="Korelasi fitur numerik & churn")

    cp = d.groupby("num_complaints").agg(rate=("churn", "mean"), n=("churn", "size")).reset_index()
    g_comp = px.bar(cp, x="num_complaints", y="rate", text_auto=".0%", template="plotly_white",
                    hover_data={"n": True}, color_discrete_sequence=[RISK_COLOR["Tinggi"]],
                    title="Churn rate berdasarkan jumlah keluhan",
                    labels={"num_complaints": "Jumlah keluhan", "rate": "Churn rate", "n": "Pelanggan"})
    g_comp.update_yaxes(tickformat=".0%")

    g_sc = px.scatter(d.sample(min(len(d), 1500), random_state=0), x="tenure_months",
                      y="monthly_charges", color="status", opacity=0.6, template="plotly_white",
                      color_discrete_map=STATUS_COLOR, title="Tenure vs tagihan bulanan (sampel ≤1.500)",
                      labels={"tenure_months": "Tenure (bulan)", "monthly_charges": "Tagihan bulanan",
                              "status": ""})

    hm = d.pivot_table(index="contract_type", columns="internet_service", values="churn", aggfunc="mean")
    g_heat = px.imshow(hm, text_auto=".0%", color_continuous_scale="Reds", aspect="auto",
                       template="plotly_white", title="Churn rate: jenis kontrak × layanan internet",
                       labels={"x": "Layanan internet", "y": "Jenis kontrak", "color": "Churn rate"})
    return kpis, insights, g_cat, g_num, g_trend, g_corr, g_comp, g_sc, g_heat


# ============ 11. CALLBACKS: PREDIKSI SINGLE ============
def risk_color(cat):
    return RISK_COLOR.get(cat, C["muted"])


@callback(Output("pred-output", "children"), Output("g-gauge", "figure"),
          Output("g-whatif-tenure", "figure"), Output("g-whatif-complaint", "figure"),
          Input("btn-predict", "n_clicks"), Input("m-single", "value"),
          State("p-region", "value"), State("p-segment", "value"), State("p-contract", "value"),
          State("p-payment", "value"), State("p-service", "value"), State("p-tenure", "value"),
          State("p-monthly", "value"), State("p-complaints", "value"), State("p-calls", "value"),
          prevent_initial_call=True)
def run_prediction(n, mkey, region, segment, contract, payment, service, tenure, monthly,
                   complaints, calls):
    if not n:                                          # ganti model sebelum pernah klik → abaikan
        raise PreventUpdate
    if None in (tenure, monthly, complaints, calls):
        msg = html.P("Lengkapi semua input numerik.", style={"color": "crimson"})
        return msg, no_update, no_update, no_update
    model, meta = MODELS[mkey]["model"], MODELS[mkey]["meta"]
    row = pd.DataFrame([{"tenure_months": tenure, "monthly_charges": monthly,
                         "total_charges": monthly * tenure, "num_complaints": complaints,
                         "support_calls": calls, "region": region, "segment": segment,
                         "contract_type": contract, "payment_method": payment,
                         "internet_service": service}])
    errs = validate_input(row, meta)
    if errs:
        return html.Div([html.P(e, style={"color": "crimson"}) for e in errs]), *[empty_fig()] * 3
    r = predict(model, row, meta).iloc[0]

    warn = [f"{LABELS[c]} = {row[c].iloc[0]:,.0f} di luar rentang data latih ({lo:,.0f}–{hi:,.0f})"
            for c, (lo, hi) in meta.get("numeric_ranges", {}).items() if not lo <= row[c].iloc[0] <= hi]
    pct = (HIST_PROBA[mkey] < r.churn_probability).mean()
    out = html.Div([
        html.Div(f"{r.churn_probability:.1%}", style={"fontSize": "40px", "fontWeight": 700,
                                                      "color": risk_color(r.risk_category)}),
        html.Div(f"Prediksi: {'Churn' if r.predicted_churn else 'Bertahan'} · Risiko {r.risk_category}",
                 style={"fontWeight": 600}),
        html.Div(f"Lebih berisiko daripada {pct:.0%} pelanggan historis · model: {MODELS[mkey]['label']}",
                 style={"color": C["muted"], "fontSize": "13px"}),
        html.P(SARAN.get(r.risk_category, ""), style={"color": C["muted"]}),
        *[html.Div("⚠ " + w, style={"color": "#B45309", "fontSize": "12px"}) for w in warn]])

    gauge = go.Figure(go.Indicator(mode="gauge+number", value=r.churn_probability * 100,
                                   number={"suffix": "%"}, gauge={
                                       "axis": {"range": [0, 100]},
                                       "bar": {"color": risk_color(r.risk_category)},
                                       "threshold": {"value": meta["decision_threshold"] * 100,
                                                     "line": {"color": C["ink"], "width": 3}},
                                       "steps": [{"range": [lo * 100, hi * 100], "color": col}
                                                 for (lo, hi), col in zip(meta["risk_bins"].values(),
                                                                          ["#E6F4EA", "#FDF0E6", "#FBE3E3"])]}))
    gauge.update_layout(margin=dict(t=20, b=10, l=20, r=20), height=240)

    def whatif(col, values, title, xlab):
        grid = pd.concat([row] * len(values), ignore_index=True)
        grid[col] = values
        grid["total_charges"] = grid["monthly_charges"] * grid["tenure_months"]
        p = predict(model, grid, meta)
        f = px.line(p, x=col, y="churn_probability", markers=True, template="plotly_white", title=title,
                    labels={col: xlab, "churn_probability": "Probabilitas churn"})
        f.add_hline(y=meta["decision_threshold"], line_dash="dash", annotation_text="threshold")
        f.add_scatter(x=[row[col].iloc[0]], y=[r.churn_probability], mode="markers", name="saat ini",
                      marker={"size": 14, "color": risk_color(r.risk_category)})
        f.update_yaxes(tickformat=".0%", range=[0, 1])
        f.update_layout(showlegend=False)
        return f

    g_ten = whatif("tenure_months", list(range(1, 73)), "What-if: bagaimana jika tenure berubah?",
                   "Tenure (bulan)")
    g_com = whatif("num_complaints", list(range(0, 11)), "What-if: bagaimana jika jumlah keluhan berubah?",
                   "Jumlah keluhan")
    return out, gauge, g_ten, g_com


# ============ 12. CALLBACKS: PREDIKSI MULTI ============
def parse_upload(contents, filename):
    """Mengubah isi dcc.Upload (base64) menjadi DataFrame. Mengembalikan (df, error)."""
    _, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    name = (filename or "").lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(io.StringIO(raw.decode("utf-8-sig"))), None
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(raw)), None
        return None, "Format file tidak didukung — gunakan .csv atau .xlsx"
    except ImportError:
        return None, "Membaca Excel butuh paket openpyxl (pip install openpyxl) — atau simpan sebagai CSV."
    except Exception as e:  # noqa: BLE001 — tampilkan pesan apa pun ke pengguna
        return None, f"Gagal membaca file: {e}"


@callback(Output("upload-store", "data"), Output("upload-status", "children"),
          Input("upload", "contents"), State("upload", "filename"), prevent_initial_call=True)
def handle_upload(contents, filename):
    def err(msgs):
        return None, html.Div([html.Div("✖ " + m, style={"color": "crimson"}) for m in msgs])

    df, e = parse_upload(contents, filename)
    if e:
        return err([e])
    if df.empty:
        return err(["File kosong."])
    df.columns = [str(c).strip() for c in df.columns]
    if "customer_id" not in df.columns:
        df.insert(0, "customer_id", [f"ROW-{i + 1:04d}" for i in range(len(df))])
    if {"total_charges", "monthly_charges", "tenure_months"} <= set(df.columns):
        df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"] * df["tenure_months"])
    errs = validate_input(df, META)
    if errs:
        return err(errs)
    n_na = int(df[META["feature_order"]].isna().any(axis=1).sum())
    note = f" · {n_na} baris punya nilai kosong (diisi otomatis oleh pipeline model)" if n_na else ""
    ok = html.Div(f"✔ {filename} — {len(df):,} baris, {df.shape[1]} kolom berhasil dibaca{note}",
                  style={"color": "#15803D", "fontWeight": 600})
    keep = [c for c in df.columns if c in TEMPLATE_COLS + ["churn"]]
    return {"filename": filename, "records": df[keep].to_dict("records")}, ok


@callback(Output("multi-results", "style"), Output("multi-kpi", "children"),
          Output("g-multi-donut", "figure"), Output("g-multi-hist", "figure"),
          Output("g-multi-region", "figure"), Output("g-multi-contract", "figure"),
          Output("multi-eval", "children"), Output("tbl-multi", "data"),
          Input("upload-store", "data"), Input("m-multi", "value"))
def update_multi(store, mkey):
    if not store:
        return {"display": "none"}, *[no_update] * 7
    model, meta = MODELS[mkey]["model"], MODELS[mkey]["meta"]
    d = predict(model, pd.DataFrame(store["records"]), meta)
    d["prediksi"] = d["predicted_churn"].map({0: "Bertahan", 1: "Churn"})
    tinggi = d.risk_category == "Tinggi"

    kpis = [kpi("Pelanggan di file", f"{len(d):,}", store["filename"]),
            kpi("Rata-rata prob. churn", f"{d.churn_probability.mean():.1%}", MODELS[mkey]["label"]),
            kpi("Diprediksi churn", f"{int(d.predicted_churn.sum()):,}",
                f"{d.predicted_churn.mean():.1%} dari file"),
            kpi("Risiko tinggi", f"{int(tinggi.sum()):,}",
                rupiah(d.loc[tinggi, "monthly_charges"].sum()) + "/bln berisiko")]

    g_donut = px.pie(d, names="risk_category", hole=0.55, color="risk_category",
                     color_discrete_map=RISK_COLOR, category_orders={"risk_category": RISK_ORDER},
                     title="Komposisi kategori risiko")
    g_donut.update_traces(textinfo="percent+label", sort=False)
    g_donut.add_annotation(text=f"<b>{len(d):,}</b><br>pelanggan", showarrow=False, font_size=14)
    g_donut.update_layout(showlegend=False)

    g_hist = px.histogram(d, x="churn_probability", nbins=25, color="risk_category",
                          color_discrete_map=RISK_COLOR, category_orders={"risk_category": RISK_ORDER},
                          template="plotly_white", title="Distribusi probabilitas churn",
                          labels={"churn_probability": "Probabilitas churn", "risk_category": "Risiko"})
    g_hist.add_vline(x=meta["decision_threshold"], line_dash="dash", annotation_text="threshold")
    g_hist.update_xaxes(range=[0, 1], tickformat=".0%")

    rr = (d.groupby(["region", "risk_category"]).size().reset_index(name="n"))
    g_region = px.bar(rr, x="n", y="region", color="risk_category", orientation="h",
                      color_discrete_map=RISK_COLOR, category_orders={"risk_category": RISK_ORDER},
                      template="plotly_white", title="Jumlah pelanggan per region & risiko",
                      labels={"n": "Pelanggan", "region": "", "risk_category": "Risiko"})

    ck = d.groupby(["contract_type", "risk_category"]).size().reset_index(name="n")
    g_contract = px.bar(ck, x="contract_type", y="n", color="risk_category",
                        color_discrete_map=RISK_COLOR,
                        category_orders={"risk_category": RISK_ORDER,
                                         "contract_type": ["Bulanan", "Tahunan", "Dua Tahun"]},
                        template="plotly_white", title="Komposisi risiko per jenis kontrak",
                        labels={"contract_type": "", "n": "Pelanggan", "risk_category": "Risiko"})

    evaluation = None
    if "churn" in d.columns and d["churn"].notna().any():     # file berlabel → evaluasi model
        e = d.dropna(subset=["churn"])
        y, p = e["churn"].astype(int), e["predicted_churn"]
        tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        cm = pd.crosstab(y.map({0: "Bertahan", 1: "Churn"}), e["prediksi"]).reindex(
            index=["Bertahan", "Churn"], columns=["Bertahan", "Churn"], fill_value=0)
        g_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", template="plotly_white",
                         labels={"x": "Prediksi", "y": "Aktual", "color": "Jumlah"},
                         title="Confusion matrix (file berisi kolom churn)")
        g_cm.update_layout(height=320, coloraxis_showscale=False)
        evaluation = html.Div(style=GRID2, children=[
            card(html.H4("Evaluasi model pada file ini", style={"marginTop": 0}),
                 html.Div(style=ROW, children=[
                     kpi("Akurasi", f"{(y == p).mean():.1%}"),
                     kpi("Precision", f"{prec:.1%}" if prec == prec else "-"),
                     kpi("Recall", f"{rec:.1%}" if rec == rec else "-")]),
                 html.Small("Muncul karena file memiliki kolom 'churn' (label aktual).",
                            style={"color": C["muted"]})),
            card(dcc.Graph(figure=g_cm))])

    rows = d.sort_values("churn_probability", ascending=False)[[i for _, i in RESULT_COLS]]
    return ({"display": "block"}, kpis, g_donut, g_hist, g_region, g_contract, evaluation,
            rows.to_dict("records"))


@callback(Output("dl-result", "data"), Input("btn-download", "n_clicks"),
          State("upload-store", "data"), State("m-multi", "value"), prevent_initial_call=True)
def download_result(_, store, mkey):
    if not store:
        raise PreventUpdate
    m = MODELS[mkey]
    d = predict(m["model"], pd.DataFrame(store["records"]), m["meta"]).sort_values(
        "churn_probability", ascending=False)
    stem = Path(store["filename"]).stem
    return dcc.send_data_frame(d.to_csv, f"prediksi_{stem}_{mkey}.csv", index=False)


@callback(Output("dl-template", "data"), Input("btn-template", "n_clicks"), prevent_initial_call=True)
def download_template(_):
    tpl = DF[TEMPLATE_COLS].head(3).copy()
    tpl["customer_id"] = ["CONTOH-1", "CONTOH-2", "CONTOH-3"]
    return dcc.send_data_frame(tpl.to_csv, "template_prediksi_churn.csv", index=False)


@callback(Output("dl-sample", "data"), Input("btn-sample", "n_clicks"), prevent_initial_call=True)
def download_sample(_):
    return dcc.send_file(str(ROOT / "data" / "new_customers.csv"))


if __name__ == "__main__":
    app.run(debug=True, port=8053)
