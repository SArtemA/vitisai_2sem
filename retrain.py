from __future__ import annotations
import os
import sqlite3
import tempfile
import traceback
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "databases" / "grapes_annotated_v2.db"
# Match MultiGrapeXGBClassifier.load_model() in models/ml_model.py.
MODEL_DIR = BASE_DIR / "models" / "trained_models_mul"
FEATURE_COLUMNS = ["elevation", "slope", "aspect", "hillshade", "mid_year_temp", "precipitation", "ndvi", "ndwi", "solar_radiation", "humidity", "wind_speed", "evapotranspiration", "evi", "lai", "land_cover_type", "soil_ph", "soil_organic_carbon", "fire_risk", "winkler_index", "fpar", "surface_pressure", "potential_evaporation_sum", "surface_sensible_heat_flux_sum", "volumetric_soil_water_layer_3", "skin_reservoir_content", "soil_temperature_level_3", "skin_temperature", "soil_bulk_density", "soil_sand", "soil_clay", "soil_texture_class"]
MUL_TARGET = ["arnsburger", "arinto", "mostosa", "abbuoto", "abouriou", "acitana"]
MODEL_FILES = {"model": "multi_output_xgb_model.joblib", "scaler": "scaler.joblib", "features": "feature_names.joblib", "targets": "target_names.joblib"}


def _micro_f1(model, scaler, x_test, y_test) -> float:
    probabilities = model.predict_proba(scaler.transform(x_test))
    positive = []
    for probability in probabilities:
        if probability.ndim != 2 or probability.shape[1] < 2:
            raise ValueError("Недостаточно классов в одной из целевых переменных для расчёта F1.")
        positive.append(probability[:, 1])
    prediction = (np.asarray(positive).T > 0.5).astype(int)
    return float(f1_score(y_test, prediction, average="micro", zero_division=0))


def _save_artifacts(model, scaler) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {"model": model, "scaler": scaler, "features": FEATURE_COLUMNS, "targets": MUL_TARGET}
    with tempfile.TemporaryDirectory(prefix="retrain_", dir=MODEL_DIR) as temp:
        temp = Path(temp)
        for key, value in artifacts.items():
            joblib.dump(value, temp / MODEL_FILES[key], compress=3)
        # Do not publish partially-written files: first write all artifacts, then replace.
        for key, filename in MODEL_FILES.items():
            os.replace(temp / filename, MODEL_DIR / filename)


def retrain_pipeline() -> dict:
    """Train a candidate; keep the active model if evaluation/comparison fails."""
    try:
        if not DB_PATH.exists():
            return {"status": "error", "message": f"База данных не найдена: {DB_PATH}"}
        with sqlite3.connect(DB_PATH) as connection:
            df = pd.read_sql_query("SELECT * FROM annotated_grapes;", connection)
        missing = [c for c in FEATURE_COLUMNS + MUL_TARGET if c not in df.columns]
        if missing:
            return {"status": "error", "message": "Отсутствуют колонки: " + ", ".join(missing)}
        if len(df) < 50:
            return {"status": "error", "message": f"Недостаточно данных: {len(df)} записей, требуется минимум 50."}

        x = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df[MUL_TARGET].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        base = XGBClassifier(colsample_bytree=0.7, gamma=0.2, learning_rate=0.1, max_depth=3,
                             n_estimators=300, reg_alpha=0.001, reg_lambda=1, subsample=0.7,
                             random_state=42, n_jobs=-1, eval_metric="logloss")
        candidate = MultiOutputClassifier(base, n_jobs=-1)
        candidate.fit(x_train_scaled, y_train)
        new_f1 = _micro_f1(candidate, scaler, x_test, y_test)

        old_model_path, old_scaler_path = MODEL_DIR / MODEL_FILES["model"], MODEL_DIR / MODEL_FILES["scaler"]
        old_exists = old_model_path.exists() and old_scaler_path.exists()
        old_f1 = None
        if old_exists:
            # If loading/evaluating the current model fails, outer handler returns error;
            # the current files are not overwritten.
            old_model, old_scaler = joblib.load(old_model_path), joblib.load(old_scaler_path)
            old_f1 = _micro_f1(old_model, old_scaler, x_test, y_test)
            if new_f1 <= old_f1:
                return {"status": "no_improvement", "old_f1": old_f1, "new_f1": new_f1,
                        "message": "Кандидат не превысил качество действующей модели."}

        _save_artifacts(candidate, scaler)
        return {"status": "success", "old_f1": old_f1, "new_f1": new_f1,
                "improved": old_f1 is not None and new_f1 > old_f1, "message": "Новая модель сохранена."}
    except Exception as exc:
        traceback.print_exc()
        return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    print(retrain_pipeline())
