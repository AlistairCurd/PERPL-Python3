"""
Test functions in io.utils with pytest.
"""

import pytest

from perpl.io import utils as ut


@pytest.mark.parametrize(
    "given, expected",
    [
        (0.0, "0"),
        (0, "0"),
        ("0", "0"),
        ("0.0", "0"),
        (1.0, "1"),
        (2.2, "2.2"),
        (" AF647 ", "AF647"),
        ("marker 1", "marker 1"),
    ],
)
def test_standardise_channel(given, expected):
    output = ut.standardise_channel(given)
    assert expected == output
