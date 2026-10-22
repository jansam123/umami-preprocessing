from __future__ import annotations

import h5py
import numpy as np
import pytest

from upp.utils.derived import H5Reader, gntau_charge_label

LABELS = [0, 0, 1, 3, 6, 2]
CHARGES = [1, -1, 1, -1, -1, 0]
EXPECTED = [0, 0, 1, 8, 10, -1]


@pytest.fixture
def jets_file(tmp_path):
    jets = np.zeros(
        6, dtype=[("GNTauUnifiedLabel", "i4"), ("TauJets_tauTruthCharge", "f4"), ("pt", "f4")]
    )
    jets["GNTauUnifiedLabel"] = LABELS
    jets["TauJets_tauTruthCharge"] = CHARGES
    jets["pt"] = np.arange(6)
    path = tmp_path / "jets.h5"
    with h5py.File(path, "w") as f:
        f["jets"] = jets
    return path


def test_gntau_charge_label(jets_file):
    with h5py.File(jets_file) as f:
        assert gntau_charge_label(f["jets"][:]).tolist() == EXPECTED


def test_stream_requested_derived_drops_added_sources(jets_file):
    reader = H5Reader(jets_file, batch_size=4, shuffle=False)
    variables = {"jets": ["pt", "GNTauChargeLabel"]}
    dtype = reader.dtypes(variables)["jets"]
    batches = list(reader.stream({"jets": ["pt", "GNTauChargeLabel"]}))
    jets = np.concatenate([b["jets"] for b in batches])
    assert jets.dtype.names == dtype.names == ("pt", "GNTauChargeLabel")
    assert jets["GNTauChargeLabel"].tolist() == EXPECTED


def test_stream_all_variables_adds_derived(jets_file):
    reader = H5Reader(jets_file, batch_size=10, shuffle=False)
    assert "GNTauChargeLabel" in reader.dtypes()["jets"].names
    jets = next(reader.stream({"jets": None}))["jets"]
    assert jets["GNTauChargeLabel"].tolist() == EXPECTED
    assert "TauJets_tauTruthCharge" in jets.dtype.names
