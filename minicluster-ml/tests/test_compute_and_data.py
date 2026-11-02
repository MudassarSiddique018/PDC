import math

import numpy as np
import pytest

from minicluster.common import compute, datasets


def test_prime_count_small_ranges():
    assert compute.prime_count(1, 101)["primes"] == 25
    assert compute.prime_count(1, 3)["primes"] == 1  # just 2
    assert compute.prime_count(1, 1_000)["largest"] == 997


def test_split_and_reduce_equals_whole():
    n = 50_000
    bounds = np.linspace(1, n + 1, 8, dtype=int)
    parts = [compute.prime_count(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
    out = compute.reduce_results("prime_count", parts)
    assert out["primes_found"] == 5133  # pi(50,000)
    assert out["numbers"] == n


def test_square_sum_matches_formula():
    n = 10_000
    parts = [compute.square_sum(1, 4000), compute.square_sum(4000, n + 1)]
    out = compute.reduce_results("square_sum", parts)
    assert int(out["sum_of_squares"]) == n * (n + 1) * (2 * n + 1) // 6
    assert out["sample"][0] == [1, 1]


def test_monte_carlo_pi_is_close():
    parts = [compute.monte_carlo_pi(i * 100_000, (i + 1) * 100_000) for i in range(3)]
    assert compute.reduce_results("monte_carlo_pi", parts)["pi_estimate"] == pytest.approx(math.pi, abs=0.02)


def test_student_dataset_is_deterministic_and_balanced():
    X1, t1 = datasets.generate_student_performance(10_000, seed=7)
    X2, t2 = datasets.generate_student_performance(10_000, seed=7)
    np.testing.assert_array_equal(X1, X2)
    assert X1.shape == (10_000, len(datasets.STUDENT_FEATURES))
    assert 0.55 < t1["passed"].mean() < 0.9
    assert set(np.unique(t1["category"])) == {0, 1, 2}


def test_csv_roundtrip_and_type_inference(tmp_path):
    path = tmp_path / "students.csv"
    datasets.write_student_csv(path, 500)
    assert datasets.load_csv(path, "passed").task == "binary"
    assert datasets.load_csv(path, "category").task == "multiclass"
    assert datasets.load_csv(path, "final_score").task == "regression"


def test_csv_string_labels_are_encoded(tmp_path):
    path = tmp_path / "iris_like.csv"
    path.write_text("a,b,species\n1,2,setosa\n3,4,virginica\n5,6,setosa\n7,8,versicolor\n")
    ds = datasets.load_csv(path, "species")
    assert ds.task == "multiclass"
    assert ds.class_names == ["setosa", "versicolor", "virginica"]
    assert ds.y.tolist() == [0, 2, 0, 1]


def test_csv_errors(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b,label\n1,x,0\n")
    with pytest.raises(ValueError):
        datasets.load_csv(path, "label")
    with pytest.raises(ValueError):
        datasets.load_csv(path, "missing")
