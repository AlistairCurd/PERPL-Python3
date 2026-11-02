"""
Tests for relative_positions.py
using pytest.
"""

from pathlib import Path

import pandas as pd
import pytest
from numpy import testing as nptest
from pandas import DataFrame
from pandas import testing as pdtest

from perpl import relative_positions as rp


@pytest.fixture(scope="module")
def input_df():
    df = pd.DataFrame(
        {
            "other": ["a", "a", "a", "a"],
            "x nm": [0, 1, 0, 1],
            "z nm": [3, 0, 0, 3],
            "chan": ["chan 0", "chan 0", "chan 1", "chan 1"],
            "y nm": [0, 2, 0, 2],
            "extra": ["b", "b", "b", "b"],
        }
    )
    return df


@pytest.fixture(scope="module")
def xyzoc_df(input_df: DataFrame):
    return input_df[["x nm", "y nm", "z nm", "other", "chan"]]


@pytest.fixture(scope="module")
def xyc_df(input_df: DataFrame):
    return input_df[["x nm", "y nm", "chan"]]


@pytest.fixture(scope="module")
def info_named_cols_arb_posns():
    info = {
        "input_path": Path("tests/data/test_xyzc_locs_named_cols_arb_posns.csv"),
        "xcol": "x nm",
        "ycol": "y nm",
        "zcol": "z nm",
        "dims": 2,
        "ccol": "chan",
        "channels_analysed": None,
    }
    return info


@pytest.fixture(scope="module")
def info_def_col_posns():
    info = {
        "input_path": Path("tests/data/test_xyzc_locs_default_col_posns.csv"),
        "xcol": None,
        "ycol": None,
        "zcol": None,
        "dims": 2,
        "ccol": None,
        "channels_analysed": None,
    }
    return info


@pytest.fixture(scope="module")
def info_no_col_names():
    info = {
        "input_path": Path("tests/data/test_xyzc_locs_no_header.csv"),
        "xcol": None,
        "ycol": None,
        "zcol": None,
        "dims": 2,
        "ccol": None,
        "channels_analysed": 2,
        "start_channel": None,
    }
    return info


def test_read_data_in_default_column_positions():
    pass


def test_read_data_in_no_header(xyzoc_df: DataFrame, info_no_col_names: dict):
    info = info_no_col_names
    data_read_in = rp.read_data_in(info)
    assert info["xcol"] == 0
    assert info["ycol"] == 1
    assert info["zcol"] is None
    assert info["ccol"] == xyzoc_df.shape[1] - 1
    assert data_read_in[info["ccol"]].to_list() == xyzoc_df["chan"].to_list()
    assert data_read_in.shape == xyzoc_df.shape
    nptest.assert_array_equal(
        data_read_in[info["ycol"]].to_numpy(), xyzoc_df["y nm"].to_numpy()
    )


def test_read_data_in_named_columns(
    input_df: DataFrame, info_named_cols_arb_posns: dict
):
    data_read_in = rp.read_data_in(info_named_cols_arb_posns)
    pdtest.assert_frame_equal(data_read_in, input_df)
    pass


def test_read_non_numeric_first_columns_without_xcol(info_def_col_posns: dict):
    info = info_def_col_posns
    with pytest.raises(SystemExit, match="gave no --xcol"):
        rp.read_data_in(info)


def test_choose_channels():
    pass
