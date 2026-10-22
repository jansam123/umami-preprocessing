"""Variables computed on read from other fields of the same group, for inputs that lack them.

| name | sources | value |
|---|---|---|
| GNTauChargeLabel | GNTauUnifiedLabel, TauJets_tauTruthCharge | 0 QCD, 1-5 / 6-10 tau+ / tau- |

``GNTauChargeLabel`` is the decay mode for positive taus, the mode + 5 for negative ones, and -1 for
taus with charge 0 or NaN.

Request them with ``derived_variables: [<name>, ...]`` in the preprocessing config.
"""

from __future__ import annotations

from collections.abc import Callable, Generator

import h5py
import numpy as np
from ftag import Cuts
from ftag.hdf5 import H5Reader as FtagH5Reader


def gntau_charge_label(jets: np.ndarray) -> np.ndarray:
    """Decay-mode label split by truth charge; the 5-prong mode (6) is folded into 3pXn (5)."""
    mode = np.minimum(jets["GNTauUnifiedLabel"].astype(np.int32), 5)
    charge = jets["TauJets_tauTruthCharge"]
    label = np.where(charge > 0, mode, np.where(charge < 0, mode + 5, -1))
    return np.where(mode == 0, 0, label).astype(np.int32)


# name -> (source fields, dtype, function of the structured array)
DERIVED: dict[str, tuple[tuple[str, ...], type, Callable[[np.ndarray], np.ndarray]]] = {
    "GNTauChargeLabel": (
        ("GNTauUnifiedLabel", "TauJets_tauTruthCharge"),
        np.int32,
        gntau_charge_label,
    ),
}


class H5Reader(FtagH5Reader):
    """ftag ``H5Reader`` that also serves the ``DERIVED`` variables missing from the files.

    A group requested as ``None`` (all variables) gets every derived variable it has sources for.
    shortcut: cuts on a derived variable are not supported, add them when a config needs one.
    """

    def _plan(self, variables: dict) -> tuple[dict, dict, dict]:
        """Return the derived names to add, fields to read and added sources to drop, per group."""
        with h5py.File(self.files[0]) as f:
            have = {g: set(f[g].dtype.names or ()) for g in variables}
        add, read, drop = {}, {}, {}
        for g, names in variables.items():
            if names is None:
                add[g] = [
                    n
                    for n, (src, *_) in DERIVED.items()
                    if n not in have[g] and set(src) <= have[g]
                ]
                read[g], drop[g] = None, []
                continue
            add[g] = [n for n in names if n in DERIVED and n not in have[g]]
            sources = [s for n in add[g] for s in DERIVED[n][0]]
            read[g] = list(dict.fromkeys([n for n in names if n not in add[g]] + sources))
            drop[g] = [s for s in sources if s not in names]
        return add, read, drop

    @staticmethod
    def _dtype(dtype: np.dtype, add: list[str], drop: list[str]) -> np.dtype:
        return np.dtype(
            [(n, dtype[n]) for n in dtype.names if n not in drop]
            + [(n, DERIVED[n][1]) for n in add]
        )

    def dtypes(self, variables: dict[str, list[str]] | None = None) -> dict[str, np.dtype]:
        if variables is None:
            with h5py.File(self.files[0]) as f:
                variables = {k: None for k in f if isinstance(f[k], h5py.Dataset)}
        add, read, drop = self._plan(variables)
        return {g: self._dtype(dt, add[g], drop[g]) for g, dt in super().dtypes(read).items()}

    def stream(
        self,
        variables: dict | None = None,
        num_global_objects: int | None = None,
        cuts: Cuts | None = None,
        start: int = 0,
        skip_batches: int = 0,
    ) -> Generator:
        if variables is None:
            variables = {self.global_objects_name: None}
        add, read, drop = self._plan(variables)
        for batch in super().stream(read, num_global_objects, cuts, start, skip_batches):
            for g, names in add.items():
                if names:
                    arr = batch[g]
                    out = np.empty(arr.shape, dtype=self._dtype(arr.dtype, names, drop[g]))
                    for n in out.dtype.names:
                        out[n] = DERIVED[n][2](arr) if n in names else arr[n]
                    batch[g] = out
            yield batch
