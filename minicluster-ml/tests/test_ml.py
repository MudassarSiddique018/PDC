import numpy as np
import pytest

from minicluster.common import ml
from minicluster.common.datasets import student_dataset
from minicluster.coordinator.jobs.training import predict, prepare


def _data(model, n=200, d=4, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = ml.add_bias(rng.normal(size=(n, d)))
    if model == ml.LINEAR:
        y = rng.normal(size=n)
    elif model == ml.LOGISTIC:
        y = rng.integers(0, 2, n)
    else:
        y = rng.integers(0, k, n)
    W = rng.normal(scale=0.3, size=(d + 1, k if model == ml.SOFTMAX else 1))
    return X, y, W


@pytest.mark.parametrize("model", [ml.LINEAR, ml.LOGISTIC, ml.SOFTMAX])
def test_gradient_matches_finite_differences(model):
    X, y, W = _data(model)
    _, grad, _ = ml.loss_and_grad_sum(model, X, y, W)
    eps = 1e-6
    num = np.zeros_like(W)
    for idx in np.ndindex(W.shape):
        Wp, Wm = W.copy(), W.copy()
        Wp[idx] += eps
        Wm[idx] -= eps
        num[idx] = (ml.loss_and_grad_sum(model, X, y, Wp)[0] - ml.loss_and_grad_sum(model, X, y, Wm)[0]) / (2 * eps)
    np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("model", [ml.LINEAR, ml.LOGISTIC, ml.SOFTMAX])
@pytest.mark.parametrize("shards", [1, 2, 3, 7])
def test_sharded_gradient_sum_equals_full_gradient(model, shards):
    """The core guarantee of gradient-sync: splitting data across PCs changes nothing."""
    X, y, W = _data(model, n=301)
    full_loss, full_grad, n = ml.loss_and_grad_sum(model, X, y, W)
    parts = [ml.loss_and_grad_sum(model, Xs, ys, W) for Xs, ys in zip(np.array_split(X, shards), np.array_split(y, shards))]
    assert sum(p[2] for p in parts) == n
    np.testing.assert_allclose(sum(p[1] for p in parts), full_grad, rtol=1e-10)
    assert sum(p[0] for p in parts) == pytest.approx(full_loss)


def test_chunked_helper_equals_single_call():
    X, y, W = _data(ml.SOFTMAX, n=1000)
    a = ml.loss_and_grad_sum(ml.SOFTMAX, X, y, W)
    b = ml.chunked_loss_and_grad(ml.SOFTMAX, X, y, W, n_chunks=4)
    np.testing.assert_allclose(a[1], b[1])
    assert a[2] == b[2]


def test_logistic_learns_student_dataset():
    ds = student_dataset(20_000, "passed")
    X_tr, y_tr, X_te, y_te, *_ = prepare(ds, 0.2, seed=1)
    W = ml.init_weights(X_tr.shape[1], 1)
    for _ in range(200):
        _, g, n = ml.loss_and_grad_sum(ml.LOGISTIC, X_tr, y_tr, W)
        W -= 0.5 * g / n
    assert ml.evaluate(ml.LOGISTIC, X_te, y_te, W)["accuracy"] > 0.85


def test_local_sgd_reduces_loss():
    ds = student_dataset(5_000, "category")
    X, y, *_ = prepare(ds, 0.1, seed=0)
    W0 = ml.init_weights(X.shape[1], 3)
    before = ml.evaluate(ml.SOFTMAX, X, y, W0)["loss"]
    W1, _ = ml.local_sgd(ml.SOFTMAX, X, y, W0, lr=0.1, batch_size=64, local_epochs=2, l2=0.0, seed=0)
    assert ml.evaluate(ml.SOFTMAX, X, y, W1)["loss"] < before * 0.7


def test_predict_uses_scaler_and_class_names():
    info = {"type": ml.LOGISTIC, "features": ["a", "b"], "mean": [10.0, 0.0], "std": [2.0, 1.0],
            "weights": [[5.0], [0.0], [0.0]], "classes": ["Fail", "Pass"]}
    out = predict(info, [{"a": 14.0, "b": 0.0}, {"a": 6.0, "b": 0.0}])
    assert [o["prediction"] for o in out] == ["Pass", "Fail"]
    assert out[0]["probabilities"]["Pass"] > 0.99
    with pytest.raises(ValueError):
        predict(info, [{"a": 1.0}])
