import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from pathlib import Path
import warnings

# Игнорируем предупреждения об устаревании для чистоты лога, если используем новые фичи
warnings.filterwarnings("ignore")


class ViticulturePipeline:
    """
    Конвейер для обучения модели бинарной классификации,
    предсказывающей пригодность для виноградарства по экологическим признакам.
    """

    def __init__(self, db_path="databases/vineyards_v3.db", model_path="xgboost_grape_model.json"):



        self.db_path = Path(Path(__file__).parent.parent, db_path)
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.median_values = None
        self.feature_columns = [
            "elevation_GEE_USGS_30m",
            "slope_GEE_USGS_30m",
            "aspect_GEE_USGS_30m",
            "hillshade_GEE_USGS_30m",
            "mid_year_temp",
            "precipitation",
            "humidity",
            "solar_radiation",
            "wind_speed",
            "evapotranspiration",
            "evi",
            "lai",
            "land_cover_type",
            "soil_ph",
            "soil_organic_carbon",
            "fire_risk",
            "winkler_index",
            "ndvi",
            "ndwi",
        ]

    def load_data(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Файл базы данных {self.db_path} не найден")

        conn = sqlite3.connect(self.db_path)
        query = f"""
        SELECT 
            {', '.join(self.feature_columns)},
            is_suitable 
        FROM vineyard_features 
        WHERE {' AND '.join([f'{col} IS NOT NULL' for col in self.feature_columns])}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Загружено {len(df)} записей из базы данных")
        return df

    def preprocess_data(self, df):

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):

        return X_train, X_test, y_train, y_test

    def train(self, X_train_raw, y_train, X_test_raw, y_test, hyperparameter_tuning=True, val_size=0.2):

        if hyperparameter_tuning:

        else:
            print("\nИспользуются параметры по умолчанию...")
            self.model = xgb.XGBClassifier(
                n_estimators=500,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                verbosity=1,
                use_label_encoder=False
            )



        print(f"\nПроизводительность модели на тестовом наборе:")
        print(f"Точность (Accuracy): {accuracy:.4f}")
        print(f"ROC AUC: {auc_score:.4f}")
        print(f"\nОтчёт классификации:")
        print(classification_report(y_test, y_pred))
        print("Матрица ошибок:")
        print(confusion_matrix(y_test, y_pred))

        return accuracy, auc_score

    def save_model(self):
        if self.model is None:
            raise ValueError("Модель не обучена")
        self.model.save_model(self.model_path)
        scaler_path = self.model_path.replace('.json', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Модель сохранена в {self.model_path}")
        print(f"Scaler сохранён в {scaler_path}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл модели {self.model_path} не найден")
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)
        scaler_path = self.model_path.replace('.json', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            print("Предупреждение: файл scaler не найден")
        print(f"Модель загружена из {self.model_path}")

    def predict(self, features):
        if self.model is None:
            self.load_model()
        X = pd.DataFrame([features], columns=self.feature_columns).astype(np.float32)
        if self.median_values is not None:
            X = X.fillna(self.median_values)
        if self.scaler is not None:
            X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        else:
            X_scaled = X
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        return bool(prediction), probability

    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("Модель ещё не обучена")
        importance = self.model.get_booster().get_score(importance_type='weight')
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['Признак', 'Важность']
        ).sort_values('Важность', ascending=False)
        return importance_df


def main():
    print("Запуск конвейера прогнозирования пригодности для виноградарства...")
    pipeline = ViticulturePipeline()
    try:
        df = pipeline.load_data()
        print(f"\nРаспределение классов:")
        print(f"Пригодные: {df['is_suitable'].sum()} (доля: {df['is_suitable'].mean():.3f})")
        unsuitable = len(df) - df['is_suitable'].sum()
        print(f"Непригодные: {unsuitable} (доля: {unsuitable / len(df):.3f})")

        X, y = pipeline.preprocess_data(df)
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y)

        accuracy, auc_score = pipeline.train(X_train, y_train, X_test, y_test, hyperparameter_tuning=True)
        pipeline.save_model()

        print(f"\nВажность признаков:")
        importance_df = pipeline.get_feature_importance()
        print(importance_df)
        print(f"\nКонвейер успешно завершён!")
    except Exception as e:
        print(f"Ошибка в конвейере: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()