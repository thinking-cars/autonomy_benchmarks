# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the helpers of the autonomy_benchmarks node.

The node itself drives the evaluation via the ``request_samples`` service of the dataset and is
covered by running it against a dataset; the parsing of the samples to evaluate is tested here,
as an unparsable value stops the node.
"""

from __future__ import annotations

import pytest
from autonomy_benchmarks.autonomy_benchmarks import parse_sample_ids


class TestParseSampleIds:
    """Tests parsing the 'sample_ids' parameter into the sample IDs to request."""

    def test_parses_comma_separated_ids(self):
        """Sample IDs are parsed in the given order."""
        assert parse_sample_ids("0,10,20") == [0, 10, 20]

    def test_parses_single_id(self):
        """A single ID is a valid request."""
        assert parse_sample_ids("7") == [7]

    def test_ignores_surrounding_whitespace(self):
        """Sample IDs separated by ', ' are parsed like IDs separated by ','."""
        assert parse_sample_ids(" 1, 2 ,3 ") == [1, 2, 3]

    @pytest.mark.parametrize("sample_ids", ["", " ", ","])
    def test_no_ids_evaluate_the_whole_dataset(self, sample_ids):
        """An empty value requests no specific samples, so the whole dataset is evaluated."""
        assert parse_sample_ids(sample_ids) == []

    @pytest.mark.parametrize("sample_ids", ["1;2", "first", "1.5", "1-2"])
    def test_rejects_values_that_are_no_ids(self, sample_ids):
        """A value that is no comma-separated list of IDs is rejected."""
        with pytest.raises(ValueError):
            parse_sample_ids(sample_ids)
