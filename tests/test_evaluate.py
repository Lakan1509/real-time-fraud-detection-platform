from ml.evaluate import evaluate_model


def test_evaluate_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = evaluate_model()

    assert result["dataset"]["total_samples"] == 3000
    assert result["dataset"]["train_samples"] == 2400
    assert result["dataset"]["test_samples"] == 600

    metrics = result["metrics"]

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0

    cm = result["confusion_matrix"]

    assert (
        cm["true_negatives"]
        + cm["false_positives"]
        + cm["false_negatives"]
        + cm["true_positives"]
        == 600
    )


def test_threshold_analysis():
    result = evaluate_model()
    thresholds = result["threshold_analysis"]

    assert set(thresholds.keys()) == {
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
    }

    for threshold_metrics in thresholds.values():
        assert 0.0 <= threshold_metrics["precision"] <= 1.0
        assert 0.0 <= threshold_metrics["recall"] <= 1.0
        assert 0.0 <= threshold_metrics["f1"] <= 1.0

    assert thresholds["0.4"]["f1"] >= thresholds["0.5"]["f1"]
