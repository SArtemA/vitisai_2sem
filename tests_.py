from pathlib import Path
import ee
from data_fetcher import *
from data_fetcher import _fetch_from_public_apis
from models.ml_model import *
from fastapi.testclient import TestClient
from fastapi import FastAPI
from main import app

class TestsClass:

    def test_ui_exists(self):
        path1 = Path(Path().cwd(), 'templates', 'base.html')
        path2 = Path(Path().cwd(), 'templates', 'map.html')
        path3 = Path(Path().cwd(), 'templates', 'predict.html')

        assert (path1.exists() and path2.exists() and path3.exists()), f"Expected files at templates were not found."


    def test_db_models_not_exists(self):

        path4 = Path(Path().cwd(), 'databases', 'vineyards_v3.db')
        path5 = Path(Path().cwd(), 'models', 'trained_models_bin').exists()
        path6 = Path(Path().cwd(), 'models', 'trained_models_mul').exists()
        ans = True
        imposter = ''
        if path4.exists():
            if path4.stat().st_size > 100000:
                ans = False
                imposter += 'vineyards_v3.db '
            else:
                print('created empty vin db')
        if path5:
            ans = False
            imposter += 'trained_models_bin '
        if path6:
            ans = False
            imposter += 'trained_models_mul '

        assert ans, f"{imposter} don't belong there"

    def test_fetch_type(self):
        assert type(fetch_environmental_data(lat=1.1, lon=1.1)) == dict, f"Expected type dict to be returned"

    def test_fetch_fallback(self, capsys):
        fetch_environmental_data(lat=1.1, lon=1.1)
        captured = capsys.readouterr()
        assert 'GEE Fetch failed' in captured.out or "пу пу пу" in captured.out

    def test_fetch_map(self, capsys):
        p_resp = _fetch_from_public_apis(1.1, 1.1)
        assert (p_resp['elevation_status'] == 'FAILED' and p_resp['slope_status']     == 'FAILED' and
                p_resp['aspect_status']    == 'FAILED' and p_resp['hillshade_status'] == 'FAILED')

    def test_both_cls_exists(self):
        try:
            from models.ml_model import BinSuitClassifier
            from models.ml_model import MultiGrapeXGBClassifier
            imp_success = True
        except Exception as e_import_class_fail:
            print(e_import_class_fail)
            imp_success = False
        assert imp_success

    def test_bin_attrs(self):
        assert hasattr(BinSuitClassifier, 'predict_suitability')
        assert hasattr(BinSuitClassifier, 'load_model')
        assert callable(getattr(BinSuitClassifier, 'predict_suitability'))
        assert callable(getattr(BinSuitClassifier, 'load_model'))

    def test_multi_attrs(self):
        assert hasattr(MultiGrapeXGBClassifier, 'predict')
        assert hasattr(MultiGrapeXGBClassifier, 'load_model')
        assert callable(getattr(MultiGrapeXGBClassifier, 'predict'))
        assert callable(getattr(MultiGrapeXGBClassifier, 'load_model'))


    def test_homepage_render(self):
        # app = FastAPI(title="Viticulture Predictor")
        client = TestClient(app)
        response = client.get("/")
        print('response.status_code', response.status_code)
        print('response.headers', response.headers)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_predict_page_render(self):
        # app = FastAPI(title="Viticulture Predictor")
        client = TestClient(app)
        response = client.get("/predict_page")
        print('response.status_code', response.status_code)
        print('response.headers', response.headers)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


    def test_predict_endpoint_validation(self):
        # app = FastAPI(title="Viticulture Predictor")
        client = TestClient(app)
        # Verify validation error occurs with missing/bad input
        response = client.post("/api/predict", json={"lat": "invalid_latitude"})
        print('response.status_code',response.status_code)
        assert response.status_code == 422