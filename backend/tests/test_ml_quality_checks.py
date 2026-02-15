import pandas as pd

from backend.ml.quality_checks import run_training_quality_checks


def _base_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"artista_id": 1, "anno": 2020, "punteggio_reale": 120.0, "f1": 1.0, "f2": 0.3},
            {"artista_id": 1, "anno": 2021, "punteggio_reale": 200.0, "f1": 1.4, "f2": 0.4},
            {"artista_id": 2, "anno": 2022, "punteggio_reale": 260.0, "f1": 1.8, "f2": 0.5},
            {"artista_id": 3, "anno": 2024, "punteggio_reale": 320.0, "f1": 2.1, "f2": 0.6},
        ]
    )


def test_quality_checks_pass_on_clean_training_frame():
    df = _base_training_frame()
    report = run_training_quality_checks(df, ["f1", "f2"], [2020, 2021, 2022, 2024])

    assert report["status"] == "pass"
    assert report["failed_checks"] == []
    assert report["summary"]["duplicate_artist_year_rows"] == 0


def test_quality_checks_fail_on_duplicate_artist_year_rows():
    df = _base_training_frame()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    report = run_training_quality_checks(df, ["f1", "f2"], [2020, 2021, 2022, 2024])

    assert report["status"] == "fail"
    assert any("duplicate_artist_year_rows" in check for check in report["failed_checks"])
