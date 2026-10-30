from datetime import date

import pytest

from cyclecast.training import calibrate, evaluate, read_archive


def test_calibration_order_statistic_and_coverage():
    assert calibrate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 10
    results = evaluate([10, 20, 30], [11, 18, 40], [9, 19, 29], 5)
    assert results["mae"] == pytest.approx(13 / 3)
    assert results["coverage"] == pytest.approx(2 / 3)
    assert results["baseline_mae"] == pytest.approx(1)


def test_archive_rejects_tampering(tmp_path):
    p = tmp_path / "data.zip"
    p.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum"):
        read_archive(p)


def test_chronology_split_boundaries():
    from cyclecast.core import split_windows

    items = [
        (date(2012, 8, 31), [], 1),
        (date(2012, 9, 1), [], 2),
        (date(2012, 10, 31), [], 3),
        (date(2012, 11, 1), [], 4),
    ]
    a, b, c = split_windows(items)
    assert [i[2] for i in a] == [1]
    assert [i[2] for i in b] == [2, 3]
    assert [i[2] for i in c] == [4]
