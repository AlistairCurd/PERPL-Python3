"""
Tests for relative_positions.py
using pytest.
"""

from pathlib import Path

import pandas as pd
import pytest
from numpy import testing as nptest
from pandas import DataFrame

from perpl import relative_positions as rp


@pytest.fixture(scope="module")
def make_input_df():
    df = pd.DataFrame(
        {
            "other": ["a", "a", "a", "a"],
            "xnm": [0, 1, 0, 1],
            "znm": [3, 0, 0, 3],
            "chan": ["chan 0", "chan 0", "chan 1", "chan 1"],
            "ynm": [0, 2, 0, 2],
            "extra": ["b", "b", "b", "b"],
        }
    )
    return df


@pytest.fixture(scope="module")
def make_xyc_df(make_input_df):
    return make_input_df[["xnm", "ynm", "chan"]]


@pytest.fixture(scope="module")
def set_up_info():
    info = {
        "input_path": Path("tests/data/test_xyzc_locs_named_cols_arb_posns.csv"),
        "xcol": "xnm",
        "ycol": "ynm",
        "zcol": "znm",
        "dims": 2,
        "ccol": "chan",
        "channels_analysed": None,
    }
    return info


def test_read_data_in_default_column_positions():
    pass


def test_read_data_in_named_columns(make_xyc_df: DataFrame, set_up_info: dict):
    data_read_in = rp.read_data_in(set_up_info)
    nptest.assert_array_equal(
        data_read_in, make_xyc_df.to_numpy()[:, 0 : set_up_info["dims"]]
    )


def test_choose_channels():
    pass
