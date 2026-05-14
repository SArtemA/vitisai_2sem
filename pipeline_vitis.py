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

    def __init__(self, db_path="vineyards_v2.db", model_path="xgboost_grape_model.json"):
        self.db_path = db_path
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.median_values = None
        self.feature_columns = [
            'elevation_GEE_USGS_30m',
            'slope_GEE_USGS_30m',
            'aspect_GEE_USGS_30m',
            'hillshade_GEE_USGS_30m',
            'mid_year_temp',
            'precipitation',
            'ndvi',
            'ndwi'
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
        X = df[self.feature_columns].copy()
        y = df['is_suitable'].astype(int)
        X = X.apply(pd.to_numeric, errors='coerce')
        # Важно: приводим к float32 для экономии памяти и скорости XGBoost
        X = X.astype(np.float32)
        print(f"Первичная обработка завершена. Размер данных: {X.shape}")
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Размер обучающей выборки: {X_train.shape[0]}")
        print(f"Размер тестовой выборки: {X_test.shape[0]}")
        print(f"Доля положительного класса в обучении: {y_train.mean():.3f}")
        return X_train, X_test, y_train, y_test

    def train(self, X_train_raw, y_train, X_test_raw, y_test, hyperparameter_tuning=True, val_size=0.2):
        # 1. Разделение на train/val внутри training set для early stopping в финале
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train_raw, y_train, test_size=val_size, random_state=42, stratify=y_train
        )

        # 2. Заполнение пропусков медианой (только на train_sub!)
        self.median_values = X_train_sub.median(numeric_only=True)

        # Функция для безопасного заполнения и масштабирования
        def prepare_data(X_raw, fit_scaler=False):
            X_filled = X_raw.fillna(self.median_values)
            if fit_scaler:
                scaled = self.scaler.fit_transform(X_filled)
            else:
                scaled = self.scaler.transform(X_filled)
            return pd.DataFrame(scaled, columns=self.feature_columns).astype(np.float32)

        # Инициализация scaler
        self.scaler = StandardScaler()

        X_train_scaled = prepare_data(X_train_sub, fit_scaler=True)
        X_val_scaled = prepare_data(X_val, fit_scaler=False)
        X_test_scaled = prepare_data(X_test_raw, fit_scaler=False)

        if hyperparameter_tuning:
            print("\nВыполняется подбор гиперпараметров...")

            # Подготовка полных данных для GridSearch (без разделения на val, т.к. будет CV)
            X_train_full_scaled = prepare_data(X_train_raw,
                                               fit_scaler=False)  # scaler уже fit на train_sub, но тут мы используем тот же scaler

            # ВАЖНО: Для GridSearchCV лучше переобучить scaler на всех данных или использовать тот же.
            # Чтобы не было data leakage, строго говоря, scaler должен быть внутри Pipeline sklearn.
            # Но для простоты оставим как есть, используя scaler, обученный на train_sub.

            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.001, 0.1],
                'reg_lambda': [1, 1.5, 2]
            }

            base_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                verbosity=0,
                # early_stopping_rounds убираем отсюда, он конфликтует в CV
                use_label_encoder=False
            )

            # РЕШЕНИЕ ПРОБЛЕМЫ: n_jobs=1 предотвращает краш на Windows
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=3,
                verbose=1,
                n_jobs=1
            )

            try:
                grid_search.fit(X_train_full_scaled, y_train)
            except Exception as e:
                print(f"Ошибка при GridSearch: {e}. Попробуем упростить сетку.")
                # Fallback: если все равно упадет, можно уменьшить сетку
                param_grid_simple = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.1]
                }
                grid_search = GridSearchCV(base_model, param_grid_simple, scoring='roc_auc', cv=3, verbose=1, n_jobs=1)
                grid_search.fit(X_train_full_scaled, y_train)

            best_params = grid_search.best_params_
            print(f"\nЛучшие параметры: {best_params}")
            print(f"Лучшая оценка кросс-валидации (ROC AUC): {grid_search.best_score_:.4f}")

            # Создаем финальную модель с лучшими параметрами
            # n_estimators берем из best_params, но можем умножить на коэффициент, если нужно больше деревьев
            final_params = best_params.copy()

            self.model = xgb.XGBClassifier(
                **final_params,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                verbosity=1,
                use_label_encoder=False
            )
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

        # ФИНАЛЬНОЕ ОБУЧЕНИЕ с Early Stopping на отложенной выборке (X_val)
        print("\nФинальное обучение модели с ранней остановкой...")
        self.model.fit(
            X_train_scaled, y_train_sub,
            eval_set=[(X_val_scaled, y_val)],
            verbose=True
        )

        # Оценка на тестовом наборе
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)

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