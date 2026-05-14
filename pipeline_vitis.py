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


class ViticulturePipeline:
    """
    Конвейер для обучения модели бинарной классификации,
    предсказывающей пригодность для виноградарства по экологическим признакам.
    """

    def __init__(self, db_path="vineyards_v2.db", model_path="xgboost_grape_model.json"):
        self.db_path = db_path
        self.model_path = model_path
        self.model = None
        self.scaler = None  # будет обучен на тренировочной выборке
        self.median_values = None  # медианы колонок для заполнения пропусков
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
        """Загружает данные из базы SQLite"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Файл базы данных {self.db_path} не найден")

        conn = sqlite3.connect(self.db_path)

        # Запрос на получение всех признаков, необходимых для обучения
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
        """
        Первичная обработка: отделение признаков и целевой переменной,
        приведение к числовому типу, замена нечисловых значений на NaN.
        """
        # Отделяем признаки и целевую переменную
        X = df[self.feature_columns].copy()
        y = df['is_suitable'].astype(int)

        # Преобразуем все признаки в числовые, нечисловые -> NaN
        X = X.apply(pd.to_numeric, errors='coerce')

        print(f"Первичная обработка завершена. Размер данных: {X.shape}")
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Разделение данных на обучающую и тестовую выборки с сохранением пропорций классов"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"Размер обучающей выборки: {X_train.shape[0]}")
        print(f"Размер тестовой выборки: {X_test.shape[0]}")
        print(f"Доля положительного класса в обучении: {y_train.mean():.3f}")

        return X_train, X_test, y_train, y_test

    def train(self, X_train_raw, y_train, X_test_raw, y_test, hyperparameter_tuning=True, val_size=0.2):
        """
        Обучение модели XGBoost с опциональной оптимизацией гиперпараметров.
        Используется дополнительная валидационная выборка для ранней остановки.
        """
        # Создаём валидационную выборку из обучающей для контроля обучения
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train_raw, y_train, test_size=val_size, random_state=42, stratify=y_train
        )

        # 1. Заполнение пропусков медианой, вычисленной только на train_sub
        self.median_values = X_train_sub.median(numeric_only=True)
        X_train_filled = X_train_sub.fillna(self.median_values)
        X_val_filled = X_val.fillna(self.median_values)
        X_test_filled = X_test_raw.fillna(self.median_values)

        # 2. Масштабирование признаков: scaler обучается на train_sub
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_filled)
        X_val_scaled = self.scaler.transform(X_val_filled)
        X_test_scaled = self.scaler.transform(X_test_filled)

        # Приводим обратно к DataFrame (важно для сохранения имён признаков)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=self.feature_columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns)

        if hyperparameter_tuning:
            print("\nВыполняется подбор гиперпараметров...")
            # Для GridSearchCV используем всю обучающую выборку (X_train_raw) с таким же заполнением и масштабированием
            X_train_full_scaled = self.scaler.transform(X_train_raw.fillna(self.median_values))
            X_train_full_scaled = pd.DataFrame(X_train_full_scaled, columns=self.feature_columns)

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
                verbosity=0
            )

            # Используем ROC AUC как метрику качества, т.к. классы могут быть несбалансированы
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=3,
                verbose=1,
                n_jobs=-1
            )

            grid_search.fit(X_train_full_scaled, y_train)

            best_params = grid_search.best_params_
            print(f"\nЛучшие параметры: {best_params}")
            print(f"Лучшая оценка кросс-валидации (ROC AUC): {grid_search.best_score_:.4f}")

            # Извлекаем лучшее количество деревьев и остальные параметры
            best_n_estimators = best_params['n_estimators']
            best_params_filtered = {k: v for k, v in best_params.items() if k != 'n_estimators'}

            # Создаём модель с лучшими параметрами и ранней остановкой
            self.model = xgb.XGBClassifier(
                **best_params_filtered,
                n_estimators=best_n_estimators,  # оптимальное количество деревьев
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                verbosity=1,
                early_stopping_rounds=20  # теперь передаётся в конструктор
            )
        else:
            print("\nИспользуются параметры по умолчанию...")
            self.model = xgb.XGBClassifier(
                n_estimators=500,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                verbosity=1,
                early_stopping_rounds=20
            )

        # Обучение с отображением прогресса на валидации
        self.model.fit(
            X_train_scaled, y_train_sub,
            eval_set=[(X_train_scaled, y_train_sub), (X_val_scaled, y_val)],
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
        """Сохраняет обученную модель и scaler"""
        if self.model is None:
            raise ValueError("Модель не обучена")
        # Сохраняем XGBoost модель
        self.model.save_model(self.model_path)

        # Сохраняем scaler
        scaler_path = self.model_path.replace('.json', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        print(f"Модель сохранена в {self.model_path}")
        print(f"Scaler сохранён в {scaler_path}")

    def load_model(self):
        """Загружает обученную модель и scaler"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл модели {self.model_path} не найден")

        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)

        # Загружаем scaler
        scaler_path = self.model_path.replace('.json', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            print("Предупреждение: файл scaler не найден, масштабирование не будет выполнено")

        print(f"Модель загружена из {self.model_path}")

    def predict(self, features):
        """Предсказание для одного образца"""
        if self.model is None:
            self.load_model()

        # Создаём DataFrame с той же структурой, что и при обучении
        X = pd.DataFrame([features], columns=self.feature_columns)

        # Заполняем пропуски теми же медианами
        if self.median_values is not None:
            X = X.fillna(self.median_values)

        # Масштабируем
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns
            )
        else:
            X_scaled = X

        # Получаем предсказание и вероятности
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]

        return bool(prediction), probability

    def get_feature_importance(self):
        """Возвращает важность признаков из обученной модели"""
        if self.model is None:
            raise ValueError("Модель ещё не обучена")

        importance = self.model.get_booster().get_score(importance_type='weight')

        # DataFrame для удобного отображения
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['Признак', 'Важность']
        ).sort_values('Важность', ascending=False)

        return importance_df


def main():
    """Главная функция запуска полного конвейера"""
    print("Запуск конвейера прогнозирования пригодности для виноградарства...")

    # Инициализация конвейера
    pipeline = ViticulturePipeline()

    try:
        # Загрузка данных
        df = pipeline.load_data()

        # Распределение классов
        print(f"\nРаспределение классов:")
        print(f"Пригодные: {df['is_suitable'].sum()} (доля: {df['is_suitable'].mean():.3f})")
        unsuitable = len(df) - df['is_suitable'].sum()
        print(f"Непригодные: {unsuitable} (доля: {unsuitable / len(df):.3f})")

        # Первичная обработка (без заполнения пропусков и масштабирования)
        X, y = pipeline.preprocess_data(df)

        # Разделение данных
        X_train, X_test, y_train, y_test = pipeline.split_data(X, y)

        # Обучение модели (с оптимизацией гиперпараметров)
        accuracy, auc_score = pipeline.train(X_train, y_train, X_test, y_test, hyperparameter_tuning=True)

        # Сохранение модели
        pipeline.save_model()

        # Вывод важности признаков
        print(f"\nВажность признаков:")
        importance_df = pipeline.get_feature_importance()
        print(importance_df)

        print(f"\nКонвейер успешно завершён!")

    except Exception as e:
        print(f"Ошибка в конвейере: {e}")
        raise


if __name__ == "__main__":
    main()