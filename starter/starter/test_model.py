
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from .ml.model import train_model, compute_model_metrics, inference

def test_train_model():

    rf = train_model(np.array([[1.0, 2.0], [2.0, 3.0]]), np.array([[1], [0]]))

    assert type(rf) == RandomForestClassifier


def test_compute_model_metrics_when_model_is_perfect():

    pr, rec, f1 = compute_model_metrics(np.array([1,1]), np.array([1,1]))

    assert pr == 1
    assert rec == 1
    assert f1 == 1


def test_inference_output_shape():
    model = RandomForestClassifier()
    model.fit(np.array([[1.0, 2.0], [2.0, 3.0]]), np.array([[1], [0]]))

    inp = np.arange(4).reshape(2, 2)

    assert len(model.predict(inp)) == 2