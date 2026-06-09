import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

import xgboost as xgb


warnings.filterwarnings("ignore")




# Locate paths relative to the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# model = None
# scaler = None
# median_values = None
# feature_columns = [
#     'elevation_GEE_USGS_30m',
#     'slope_GEE_USGS_30m',
#     'aspect_GEE_USGS_30m',
#     'hillshade_GEE_USGS_30m',
#     'mid_year_temp', 'precipitation', 'ndvi', 'ndwi',
#     'solar_radiation', 'humidity', 'wind_speed',
#     'evapotranspiration', 'evi', 'lai', 'soil_ph',
#     'soil_organic_carbon', 'fire_risk'
# ]
#

# def load_model_if_exists():
#     global model, scaler, median_values
#     if not os.path.exists(MODEL_PATH):
#         print(f"Файл модели {MODEL_PATH} не найден.")
#         return
#
#     model = xgb.XGBClassifier()
#     model.load_model(MODEL_PATH)
#
#     if os.path.exists(SCALER_PATH):
#         scaler = joblib.load(SCALER_PATH)
#     else:
#         raise FileNotFoundError(f"Scaler {SCALER_PATH} не найден, модель неполная.")
#
#     median_values = None
#     print("Модель и scaler загружены.")


# def predict_suitability(env_data: dict) -> bool:
#     global model, scaler, median_values
#     if model is None:
#         raise RuntimeError("Модель не загружена.")
#
#     X = pd.DataFrame([env_data], columns=feature_columns)
#
#     if median_values is not None:
#         X = X.fillna(median_values)
#
#     if scaler is not None:
#         X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
#     else:
#         X_scaled = X
#
#     pred = model.predict(X_scaled)[0]
#     return bool(pred)

FEATURES = ['elevation', 'slope', 'aspect', 'hillshade', 'mid_year_temp', 'precipitation', 'ndvi', 'ndwi',
            'solar_radiation', 'humidity', 'wind_speed', 'evapotranspiration', 'evi', 'lai', 'land_cover_type',
            'soil_ph', 'soil_organic_carbon', 'fire_risk', 'winkler_index']

MUL_TARGET = ['arnsburger', 'arinto', 'mostosa', 'abbuoto', 'abouriou', 'acitana']
BIN_TARGET = 'is_suitable'


class BinSuitClassifier:
    model = None
    scaler = None

    def __init__(self, path_to_model=None, path_to_save=None):
        self.median_values = None
        self.path_to_save = path_to_save
        print('1 load_model or train model dataset in pd.DataFrame format needed')
        self.load_model(path_to_model)
        print('2 predict')

    def load_model(self, path_to_model=None):

        print("\nLoading the trained model, scaler, and metadata...")
        if path_to_model:
            model_dir = path_to_model
        else:
            model_dir = os.path.join(BASE_DIR, 'trained_models_bin')

        model_filename = os.path.join(model_dir, 'xgboost_grape_model.json')
        scaler_filename = os.path.join(model_dir, 'xgboost_grape_model_scaler.pkl')

        try:
            self.model = joblib.load(model_filename)
            self.scaler = joblib.load(scaler_filename)

        except FileNotFoundError:
            print(f"Error: Model files not found in {model_dir}. Please ensure the training script was run.")
            exit()

    def save_model(self, bin_model, scaler_bin):
        print("\nSaving the trained model...")
        model_dir = os.path.join(BASE_DIR, 'trained_models_bin')

        os.makedirs(model_dir, exist_ok=True)
        model_filename = os.path.join(model_dir, 'xgboost_bin_model.json')
        scaler_filename = os.path.join(model_dir, 'xgboost_bin_model_scaler.pkl')

        joblib.dump(bin_model, model_filename)
        joblib.dump(scaler_bin, scaler_filename)

        print(f"Модель сохранена в {model_filename}")
        print(f"Scaler сохранён в {scaler_filename}")

    def train(self, df):
        X = df[FEATURES].copy()
        y = df[BIN_TARGET].astype(int)
        X = X.apply(pd.to_numeric, errors='coerce')
        # Важно: приводим к float32 для экономии памяти и скорости XGBoost
        X = X.astype(np.float32)
        print(f"Первичная обработка завершена. Размер данных: {X.shape}")

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Размер обучающей выборки: {X_train_raw.shape[0]}")
        print(f"Размер тестовой выборки: {X_test_raw.shape[0]}")
        print(f"Доля положительного класса в обучении: {y_train.mean():.3f}")

        # 1. Разделение на train/val внутри training set для early stopping в финале
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train_raw, y_train, test_size=0.15, random_state=42, stratify=y_train
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
            return pd.DataFrame(scaled, columns=FEATURES).astype(np.float32)

        # Инициализация scaler
        self.scaler = StandardScaler()

        X_train_scaled = prepare_data(X_train_sub, fit_scaler=True)
        X_val_scaled = prepare_data(X_val, fit_scaler=False)
        X_test_scaled = prepare_data(X_test_raw, fit_scaler=False)

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

        # ФИНАЛЬНОЕ ОБУЧЕНИЕ с Early Stopping на отложенной выборке (X_val)
        print("\nФинальное обучение модели с ранней остановкой...")
        self.model.fit(
            X_train_scaled, y_train_sub,
            eval_set=[(X_val_scaled, y_val)],
            verbose=True
        )

        # Оценка на тестовом наборе

        # y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        print("Training complete.")
        print("\nEvaluating the model...")

        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred_proba_reshaped = np.array(y_pred_proba).T
        y_pred_binary = (y_pred_proba_reshaped > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        f1_micro = f1_score(y_test, y_pred_binary, average='micro')
        f1_macro = f1_score(y_test, y_pred_binary, average='macro')

        auc_score = roc_auc_score(y_test, y_pred_proba)

        print(f"\n--- Evaluation Results ---")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1 Score (Micro): {f1_micro:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")

        print(f"ROC AUC Score (Macro Avg): {auc_score:.4f}")

        print("\nTrain finished.")

        self.save_model(self.model, self.scaler)

    def predict_suitability(self, env_data: dict) -> bool:
        # global model, scaler, median_values
        if self.model is None:
            raise RuntimeError("Модель не загружена.")

        X = pd.DataFrame([env_data], columns=FEATURES)

        if self.scaler is not None:
            X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        else:
            X_scaled = X

        pred = self.model.predict(X_scaled)[0]
        return bool(pred)


class MultiGrapeXGBClassifier:
    loaded_model = None
    loaded_scaler = None
    loaded_feature_names = None
    loaded_target_names = None

    def __init__(self, path_to_model=None, path_to_save=None):
        self.path_to_save = path_to_save
        print('1 load_model or train model dataset in pd.DataFrame format needed')
        self.load_model(path_to_model)
        print('2 predict')

    def load_model(self, path_to_model=None):
        print("\nLoading the trained model, scaler, and metadata...")
        if path_to_model:
            model_dir = path_to_model
        else:
            model_dir = os.path.join(BASE_DIR, 'trained_models_multi')

        model_filename = os.path.join(model_dir, 'multi_output_xgb_model.joblib')
        scaler_filename = os.path.join(model_dir, 'scaler.joblib')
        feature_names_filename = os.path.join(model_dir, 'feature_names.joblib')
        target_names_filename = os.path.join(model_dir, 'target_names.joblib')

        try:
            self.loaded_model = joblib.load(model_filename)
            self.loaded_scaler = joblib.load(scaler_filename)
            self.loaded_feature_names = joblib.load(feature_names_filename)
            self.loaded_target_names = joblib.load(target_names_filename)
            print("Model, scaler, and metadata loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model files not found in {model_dir}. Please ensure the training script was run.")
            exit()

    def save_model(self, multi_output_xgb_model, scaler_multi, feature_names, target_names):
        print("\nSaving the trained model...")
        model_dir = self.path_to_save if self.path_to_save else os.path.join(BASE_DIR, 'trained_models')
        model_filename = os.path.join(model_dir, 'multi_output_xgb_model.joblib')

        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(multi_output_xgb_model, model_filename)
        print(f"Model saved successfully to {model_filename}")

        scaler_filename = os.path.join(model_dir, 'scaler.joblib')
        joblib.dump(scaler_multi, scaler_filename)
        print(f"Scaler saved successfully to {scaler_filename}")

        feature_names_filename = os.path.join(model_dir, 'feature_names.joblib')
        joblib.dump(feature_names, feature_names_filename)
        print(f"Feature names saved successfully to {feature_names_filename}")

        target_names_filename = os.path.join(model_dir, 'target_names.joblib')
        joblib.dump(target_names, target_names_filename)
        print(f"Target names saved successfully to {target_names_filename}")

    def train(self, df):
        feature_names = FEATURES
        target_names = MUL_TARGET  # ['Arnsburger', 'Arinto', 'Mostosa', 'Abbuoto', 'Abouriou', 'Acitana']

        X = df[feature_names].apply(pd.to_numeric)
        y = df[target_names].apply(pd.to_numeric)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        print("\nScaling features...")
        scaler_multi = StandardScaler()
        X_train_scaled = scaler_multi.fit_transform(X_train)
        X_test_scaled = scaler_multi.transform(X_test)

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)

        print("\nInitializing base XGBoost MultiOutputClassifier...")
        xgb_base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )

        multi_output_xgb_model = MultiOutputClassifier(xgb_base_model, n_jobs=-1)

        # Define parameter grid for GridSearchCV
        # Prefix parameters with 'estimator__' to pass them to the XGBClassifier
        param_grid = {
            'estimator__max_depth': [3, 5, 7],
            'estimator__learning_rate': [0.01, 0.05, 0.1],
            'estimator__n_estimators': [100, 200, 300],
            'estimator__subsample': [0.7, 0.8, 1.0],
            'estimator__colsample_bytree': [0.7, 0.8, 1.0],
            'estimator__gamma': [0, 0.1, 0.2],
            'estimator__reg_alpha': [0, 0.001, 0.1],
            'estimator__reg_lambda': [1, 1.5, 2]
        }

        print("Starting Grid Search CV...")
        # Using micro-averaged F1 score as the optimization metric
        grid_search = GridSearchCV(
            estimator=multi_output_xgb_model,
            param_grid=param_grid,
            scoring='f1_micro',
            cv=3,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train_scaled_df, y_train)

        print("\nGrid Search completed.")
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best CV F1 Score (Micro): {grid_search.best_score_:.4f}")

        # Retrieve the best estimator found
        best_multi_output_model = grid_search.best_estimator_

        print("\nEvaluating the best model...")
        y_pred_proba = best_multi_output_model.predict_proba(X_test_scaled_df)
        y_pred_proba_reshaped = np.array([prob[:, 1] for prob in y_pred_proba]).T
        y_pred_binary = (y_pred_proba_reshaped > 0.5).astype(int)

        f1_micro = f1_score(y_test, y_pred_binary, average='micro')
        f1_macro = f1_score(y_test, y_pred_binary, average='macro')
        jaccard = jaccard_score(y_test, y_pred_binary, average='samples')

        roc_auc_scores = []
        for i, target_name in enumerate(target_names):
            if len(np.unique(y_test.iloc[:, i])) > 1:
                roc_auc_scores.append(roc_auc_score(y_test.iloc[:, i], y_pred_proba_reshaped[:, i]))
            else:
                print(f"Skipping ROC AUC for {target_name} as it has only one class in y_test.")

        roc_auc_macro = np.mean(roc_auc_scores) if roc_auc_scores else np.nan

        print(f"\n--- Evaluation Results ---")
        print(f"F1 Score (Micro): {f1_micro:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"Jaccard Score (Samples Avg): {jaccard:.4f}")
        print(f"ROC AUC Score (Macro Avg): {roc_auc_macro:.4f}")

        print("\nPer-label F1 Scores:")
        f1_per_label = f1_score(y_test, y_pred_binary, average=None)
        for i, target_name in enumerate(target_names):
            print(f"  {target_name}: {f1_per_label[i]:.4f}")

        print("\nTrain finished.")
        print('\nSaving model')
        self.save_model(best_multi_output_model, scaler_multi, feature_names, target_names)

    def predict(self, new_data_raw):
        if not self.loaded_model:
            print('model not loaded')
            return None
        if not self.loaded_scaler:
            print('scaler not loaded')
            return None
        if not self.loaded_feature_names:
            print('feature names not loaded')
            return None
        if not self.loaded_target_names:
            print('target names not loaded')
            return None

        print("\nMaking predictions with the loaded model and ranking results...")

        new_data_raw = pd.DataFrame([new_data_raw], columns=self.loaded_feature_names)

        new_data_scaled = self.loaded_scaler.transform(new_data_raw)
        new_data_scaled_df = pd.DataFrame(new_data_scaled, columns=self.loaded_feature_names)

        sample_pred_probas = self.loaded_model.predict_proba(new_data_scaled_df)
        probabilities_for_targets = [float(p[0][1]) for p in sample_pred_probas]

        target_probs_dict = dict(zip(self.loaded_target_names, probabilities_for_targets))
        ranked_targets = sorted(target_probs_dict.items(), key=lambda item: item[1], reverse=True)

        rankings = []
        for tup_grape in ranked_targets:
            rankings.append({"grape": tup_grape[0], "score": round(tup_grape[1] * 100, 2)})

        return rankings