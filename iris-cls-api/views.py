import json
import os
import pickle

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db import connection

# Resolve model path relative to project root (sample/) where api/ lives
MODEL_REL_PATH = os.path.join(settings.BASE_DIR, 'resources', 'models', 'model_cls_iris_v1.0.pkl')

# Khai báo mô hình cần load, giả sử bắt đầu là None
_model = None

# Khai báo thêm biến để giữa tên bảng
# ---- Simple SQLite logging (4 fields) using Django's DB connection ----
TABLE_NAME = 'iris_table'


def ensure_table_exists():
    """Create the samples table with exactly 4 REAL feature columns and 1 target if it does not exist.
    -features tương ứng sepal_length,sepal_width,petal_length,petal_width
    -taget là species
    """
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                sepal_length REAL NOT NULL,
                sepal_width REAL NOT NULL,
                petal_length REAL NOT NULL,
                petal_width REAL NOT NULL,
                species text NOT NULL
            )
            """
        )


def save_sample(features, prediction):
    """Persist a single 4-feature sample anmd 1 target to SQLite.
    features: list[float] of length 4
    target: str
    """
    if not isinstance(features, list) or len(features) != 4:
        raise ValueError('features must be a list of 4 numeric values')

    ensure_table_exists()

    sql = (
        f"INSERT INTO {TABLE_NAME} ("
        f"sepal_length,sepal_width,petal_length,petal_width, species) VALUES (%s, %s,%s, %s,%s)"
    )
    params = features + [prediction]
    with connection.cursor() as cursor:
        cursor.execute(sql, params)


def get_model():
    """
    Nạp model để chọn trước đó
    Load model lazily"""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_REL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_REL_PATH}")
        with open(MODEL_REL_PATH, 'rb') as f:
            _model = pickle.load(f)
    return _model


@csrf_exempt
def predict_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST is allowed'}, status=405)

    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        return JsonResponse({'error': 'Invalid JSON body'}, status=400)

    features = data.get('features')
    if not isinstance(features, list):
        return JsonResponse({'error': "Request JSON must include a 'features' array"}, status=400)

    if len(features) != 4:
        return JsonResponse({'error': 'Expected 4 feature values in the features array'}, status=400)

    try:
        # convert to floats
        x = [float(v) for v in features]
    except Exception:
        return JsonResponse({'error': 'All feature values must be numeric'}, status=400)

    try:
        model = get_model()
        # scikit-learn expects 2D array for a single sample
        pred = model.predict([x])
        # Ensure JSON serializable
        prediction_value = pred[0]
        # Save the 4 input features to SQLite when prediction succeeds
        saved = True
        try:
            save_sample(x, prediction_value)
        except Exception as e:
            # Do not fail the request if logging fails; just report status
            saved = False
        return JsonResponse({'prediction': prediction_value, 'saved': saved})
    except FileNotFoundError as e:
        return JsonResponse({'error': str(e)}, status=500)
    except Exception as e:
        return JsonResponse({'error': f'Prediction failed: {e.__class__.__name__}: {e}'}, status=500)
