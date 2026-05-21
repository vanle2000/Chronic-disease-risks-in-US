"""Unit tests for CDC CDI modeling utilities."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from src.models.train import build_state_disease_matrix, cluster_states


def _make_state_df(n_states: int = 20, n_records: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    topics = ["Cancer", "Diabetes", "Asthma", "Cardiovascular Disease", "Mental Health"]
    states = [f"S{i:02d}" for i in range(n_states)]

    # Create one-hot topic columns
    rows = []
    for _ in range(n_records):
        topic = rng.choice(topics)
        row = {f"topic_{t}": int(t == topic) for t in topics}
        row["LocationAbbr"] = rng.choice(states)
        rows.append(row)

    return pd.DataFrame(rows)


class TestStateDiseaseMatrix:
    def test_output_shape(self):
        df = _make_state_df(n_states=10)
        matrix = build_state_disease_matrix(df)
        assert matrix.shape[0] == 10  # one row per state
        topic_cols = [c for c in df.columns if c.startswith("topic_")]
        assert matrix.shape[1] == len(topic_cols)

    def test_rows_sum_to_one(self):
        df = _make_state_df(n_states=15)
        matrix = build_state_disease_matrix(df)
        row_sums = matrix.sum(axis=1)
        assert (abs(row_sums - 1.0) < 1e-9).all(), "Rows should sum to 1 after normalization"

    def test_no_negative_values(self):
        df = _make_state_df()
        matrix = build_state_disease_matrix(df)
        assert (matrix >= 0).all().all()

    def test_index_is_state_abbr(self):
        df = _make_state_df(n_states=8)
        matrix = build_state_disease_matrix(df)
        assert matrix.index.name == "LocationAbbr"


class TestClusterStates:
    def _make_matrix(self, n_states: int = 30) -> pd.DataFrame:
        df = _make_state_df(n_states=n_states, n_records=1000)
        return build_state_disease_matrix(df)

    def test_returns_silhouette_score(self):
        matrix = self._make_matrix()
        result = cluster_states(matrix, n_clusters=3)
        assert "silhouette" in result
        assert -1.0 <= result["silhouette"] <= 1.0

    def test_cluster_column_in_output(self):
        matrix = self._make_matrix()
        result = cluster_states(matrix, n_clusters=3)
        assert "cluster" in result["state_clusters"].columns

    def test_n_distinct_clusters(self):
        matrix = self._make_matrix(n_states=30)
        result = cluster_states(matrix, n_clusters=3)
        n_unique = result["state_clusters"]["cluster"].nunique()
        assert n_unique == 3

    def test_cluster_labels_in_range(self):
        matrix = self._make_matrix()
        result = cluster_states(matrix, n_clusters=4)
        labels = result["state_clusters"]["cluster"]
        assert labels.min() >= 0
        assert labels.max() <= 3

    def test_silhouette_improves_with_good_structure(self):
        """
        With clearly separated groups, silhouette should be above 0.
        This test verifies the clustering is actually learning structure.
        """
        rng = np.random.default_rng(0)
        # Create 3 clearly separated clusters
        n_per = 10
        g1 = pd.DataFrame({"topic_A": 0.9, "topic_B": 0.05, "topic_C": 0.05}, index=range(n_per))
        g2 = pd.DataFrame({"topic_A": 0.05, "topic_B": 0.9, "topic_C": 0.05}, index=range(n_per, 2 * n_per))
        g3 = pd.DataFrame({"topic_A": 0.05, "topic_B": 0.05, "topic_C": 0.9}, index=range(2 * n_per, 3 * n_per))
        matrix = pd.concat([g1, g2, g3])
        matrix.index.name = "LocationAbbr"

        result = cluster_states(matrix, n_clusters=3)
        assert result["silhouette"] > 0.5, (
            f"Expected high silhouette on clearly separated data, got {result['silhouette']:.3f}"
        )
