# sesi4/cek_inference.py — Cek model churn sebelum dipakai di aplikasi Streamlit
# Jalankan dari folder dashboard_course:  python sesi4/cek_inference.py
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
DATA_BARU = ROOT / "data" / "new_customers.csv"


def kategori_risiko(p, bins):
    for nama, (lo, hi) in bins.items():
        if lo <= p < hi or (hi == 1.0 and p == 1.0):
            return nama
    return "-"


# 1. Pastikan file model ada
for f in (MODEL_PATH, META_PATH):
    if not f.exists():
        sys.exit(f"File tidak ditemukan: {f}\nPastikan folder 'models' ada di dalam dashboard_course.")

meta = json.loads(META_PATH.read_text(encoding="utf-8"))
model = joblib.load(MODEL_PATH)
FITUR = meta["feature_order"]

print("=" * 60)
print(f"Model     : {meta['model_name']} v{meta['model_version']}")
print(f"Algoritma : {meta['algorithm']}")
print(f"Metrik    : ROC-AUC {meta['holdout_metrics']['roc_auc']}, "
      f"akurasi {meta['holdout_metrics']['accuracy']}, F1 {meta['holdout_metrics']['f1']}")

# 2. Cek versi scikit-learn (beda versi bisa membuat model gagal dimuat atau hasilnya beda)
if sklearn.__version__ != meta["sklearn_version"]:
    print(f"PERINGATAN: scikit-learn terpasang {sklearn.__version__}, "
          f"model dibuat dengan {meta['sklearn_version']}.")
else:
    print(f"scikit-learn {sklearn.__version__} cocok dengan versi model.")
print("=" * 60)

# 3. Prediksi dua contoh pelanggan buatan
contoh = pd.DataFrame([
    dict(tenure_months=3, monthly_charges=700_000, total_charges=2_100_000,
         num_complaints=4, support_calls=6, region="Jawa", segment="Retail",
         contract_type="Bulanan", payment_method="E-Wallet", internet_service="DSL"),
    dict(tenure_months=60, monthly_charges=400_000, total_charges=24_000_000,
         num_complaints=0, support_calls=1, region="Sumatera", segment="Korporat",
         contract_type="Dua Tahun", payment_method="Kartu Kredit", internet_service="Fiber"),
])[FITUR]

proba = model.predict_proba(contoh)[:, 1]
print("\nContoh prediksi:")
for label, p in zip(["Pelanggan baru, kontrak bulanan, banyak keluhan",
                     "Pelanggan lama, kontrak dua tahun, tanpa keluhan"], proba):
    print(f"  - {label}: {p:.1%} -> risiko {kategori_risiko(p, meta['risk_bins'])}")

# 4. Prediksi data pelanggan baru (jika ada)
if DATA_BARU.exists():
    df = pd.read_csv(DATA_BARU)
    kurang = [c for c in FITUR if c not in df.columns]
    if kurang:
        print(f"\n{DATA_BARU.name} tidak punya kolom: {', '.join(kurang)}")
    else:
        df["proba_churn"] = model.predict_proba(df[FITUR])[:, 1]
        df["prediksi"] = (df["proba_churn"] >= meta["decision_threshold"]).astype(int)
        df["risiko"] = df["proba_churn"].apply(lambda p: kategori_risiko(p, meta["risk_bins"]))
        print(f"\n{DATA_BARU.name}: {len(df):,} pelanggan diprediksi")
        print(df["risiko"].value_counts().reindex(meta["risk_bins"].keys(), fill_value=0)
                .to_string())
        kolom = [c for c in ["customer_id", "contract_type", "tenure_months"] if c in df.columns]
        print("\n5 pelanggan dengan risiko tertinggi:")
        print(df.sort_values("proba_churn", ascending=False)[kolom + ["proba_churn", "risiko"]]
                .head(5).to_string(index=False, formatters={"proba_churn": "{:.1%}".format}))
else:
    print(f"\n{DATA_BARU.name} tidak ditemukan, langkah prediksi massal dilewati.")

print("\nModel siap dipakai. Lanjut: streamlit run sesi4/app_prediksi.py")