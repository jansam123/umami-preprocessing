from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar
import numpy as np
from ftag.transform import Transform


Batch = dict[str, np.ndarray]
    
@dataclass
class SubjetTransform(Transform):

    def __call__(self, batch: Batch) -> Batch:
        batch = super().__call__(batch)
        if "truth_subjets" in batch:
            batch["truth_subjets"] = self._transform_subjets(batch["truth_subjets"], batch["jets"])
            batch["jets"] = self._add_subjet_fields_to_jets(batch["jets"], batch["truth_subjets"])
        if "jets" in batch:
            batch["jets"] = self._add_truth_total_fields_to_jets(batch["jets"])
            batch["jets"] = self._add_is_two_subjet_to_jets(batch["jets"])
        return batch

    # R10TruthLabel_R22v1 values whose "total" truth quantities come from the tau
    # truth system: 16 = HtautauHad, 19 = ZtautauHad. Any other jet (in this
    # workflow, QCD) falls back to the large-R truth jet quantity.
    _TAU_TRUTH_LABELS: ClassVar[tuple[int, ...]] = (16, 19)

    # Flavour-dependent "total" truth quantities added to the jets group.
    # Maps the output field -> (source for tau-truth jets, source for all other jets).
    _TRUTH_TOTAL_FIELDS: ClassVar[dict[str, tuple[str, str]]] = {
        "TruthTotalMass": ("TauTruthTotalMass", "R10WZTruthLabel_R22v1_MatchedTruthJetMass"),
        "TruthTotalPt":   ("TauTruthTotalPt",   "R10WZTruthLabel_R22v1_MatchedTruthJetPt"),
    }

    # Derived binary jets flags. Maps the output field -> the source field it is
    # computed from. is_two_subjet = 1 if n_TruthSubjets > 1 else 0. Jets with
    # n_TruthSubjets == 0 are expected to be removed upstream by a config cut.
    _DERIVED_SCALAR_FIELDS: ClassVar[dict[str, str]] = {
        "is_two_subjet": "n_TruthSubjets",
    }

    # Output fields that don't exist in the source HDF5, per group.
    # Maps each computed output field to a source field that will be loaded anyway.
    # Used by get_dtype (inverse=True) so the reader knows what to load.
    _COMPUTED_OUTPUT_TO_SOURCE: ClassVar[dict[str, dict[str, str]]] = {
        "truth_subjets": {
            "log_pt":           "pt",
            "log_pt_ptjet":     "pt",
            "cos_dphi":         "phi",
            "sin_dphi":         "phi",
            "log_mass":         "mass",
            "log_mass_massjet": "mass",
        },
        "jets": {
            # All leading_/subleading_subjet_* fields are computed — map to a
            # source field that will already be requested (pt is always in inputs).
            **{f"{lead}_subjet_{var}": "pt"
               for lead in ("leading", "subleading")
               for var in ("log_pt", "pt", "log_pt_ptjet", "mass", "log_mass", "log_mass_massjet", "eta", "phi", "energy", "dphi", "deta", "cos_dphi", "sin_dphi")},
        },
    }

    def map_variable_names(
        self,
        name: str,
        variables: list[str],
        inverse: bool = False,
    ) -> list[str]:
        """Translate variable names between source and output space for computed fields.

        When inverse=True (reader asking "what source field do I load?"):
        - For truth_subjets: maps computed output fields (cos_dphi, sin_dphi,
          log_pt_ptjet, log_mass, log_mass_massjet) back to a real source field
          so the reader loads it, deduplicating as needed.
        - For jets: all leading_/subleading_subjet_* fields are purely computed
          (not present in source), so they are stripped from the variable list.
          The flavour-dependent TruthTotalMass/TruthTotalPt fields are likewise
          stripped, but their real source dependencies are injected so the reader
          loads them even when they are not listed explicitly in the config.
        When inverse=False, applies the base-class renaming logic only.
        """
        key = name.lstrip("/")
        computed = set(self._COMPUTED_OUTPUT_TO_SOURCE.get(key, {}))
        if key == "jets":
            computed |= set(self._TRUTH_TOTAL_FIELDS)
            computed |= set(self._DERIVED_SCALAR_FIELDS)
        if not computed or not inverse:
            return super().map_variable_names(name, variables, inverse=inverse)

        if key == "jets":
            # Computed subjet fields don't exist in source — drop them entirely.
            result = [v for v in variables if v not in computed]
            # If a TruthTotal field is requested, make sure the source fields it is
            # derived from are loaded (the flavour label + both per-flavour inputs).
            if any(f in variables for f in self._TRUTH_TOTAL_FIELDS):
                deps = ["R10TruthLabel_R22v1"]
                for htt_src, other_src in self._TRUTH_TOTAL_FIELDS.values():
                    deps += [htt_src, other_src]
                for dep in deps:
                    if dep not in result:
                        result.append(dep)
            # Derived binary flags need their single source field loaded.
            for out_field, src in self._DERIVED_SCALAR_FIELDS.items():
                if out_field in variables and src not in result:
                    result.append(src)
            return result

        # truth_subjets: map each computed field back to its real source field,
        # deduplicating so the reader doesn't request the same field twice.
        source_map = self._COMPUTED_OUTPUT_TO_SOURCE[key]
        seen: set[str] = set()
        result: list[str] = []
        for var in variables:
            src = source_map.get(var, var)
            if src not in seen:
                seen.add(src)
                result.append(src)
        return result

    def map_dtype(self, name: str, dtype: np.dtype) -> np.dtype:
        """Return the post-transform dtype, used by the writer to set up datasets.

        Overrides Transform.map_dtype to account for the jets extension added by
        _add_subjet_fields_to_jets. truth_subjets is fully reconstructed by
        _transform_subjets after map_variables runs, so its dtype is left unchanged
        here to avoid a cast error during map_variables.
        """
        # First apply any variable renaming from the base class
        dtype = super().map_dtype(name, dtype)

        key = name.lstrip("/")

        if key == "jets":
            subjet_vars = ["log_pt", "pt", "log_pt_ptjet", "mass", "log_mass", "log_mass_massjet", "eta", "energy", "phi", "cos_dphi", "sin_dphi", "deta", "dphi"]
            leads = ["leading", "subleading"]
            existing = {f for f, *_ in dtype.descr}
            new_fields = [
                (f"{lead}_subjet_{var}", np.float32)
                for lead in leads for var in subjet_vars
                if f"{lead}_subjet_{var}" not in existing
            ]
            new_fields += [
                (f, np.float32) for f in self._TRUTH_TOTAL_FIELDS if f not in existing
            ]
            new_fields += [
                (f, np.int8) for f in self._DERIVED_SCALAR_FIELDS if f not in existing
            ]
            return np.dtype(dtype.descr + new_fields)

        return dtype

    def output_dtype(self, name: str, dtype: np.dtype) -> np.dtype:
        """Return the final output dtype including truth_subjets transformation.

        Used by setup_writer to create HDF5 datasets with the correct structure.
        """
        key = name.lstrip("/")

        if key == "truth_subjets":
            return np.dtype([
                ("log_pt",           np.float32),
                ("pt",               np.float32),
                ("log_pt_ptjet",     np.float32),
                ("mass",             np.float32),
                ("log_mass",         np.float32),
                ("log_mass_massjet", np.float32),
                ("eta",              np.float32),
                ("energy",           np.float32),
                ("phi",              np.float32),
                ("valid",            bool),
                ("dphi",             np.float32),
                ("cos_dphi",         np.float32),
                ("sin_dphi",         np.float32),
                ("deta",             np.float32),
            ])

        return self.map_dtype(name, dtype)

    def _transform_subjets(self, subjets: np.ndarray, jets: np.ndarray) -> np.ndarray:
        """Apply log transforms and add cos_dphi/sin_dphi fields to truth_subjets."""
        valid = subjets["valid"]
        valid_mass = valid & (subjets["mass"] > 0)

        pt               = subjets["pt"] * 1e-3
        log_pt           = np.where(valid,      np.log(subjets["pt"] * 1e-3),                              0).astype(np.float32)
        log_pt_ptjet     = np.where(valid,      np.log(subjets["pt"] / jets["pt"][:, np.newaxis]),         0).astype(np.float32)
        mass             = np.where(valid_mass, subjets["mass"] * 1e-3,                                    0).astype(np.float32)
        log_mass         = np.where(valid_mass, np.log(subjets["mass"] * 1e-3),                            0).astype(np.float32)
        log_mass_massjet = np.where(valid_mass, np.log(subjets["mass"] / jets["mass"][:, np.newaxis]),     0).astype(np.float32)
        eta    = np.where(valid,      subjets["eta"],    0).astype(np.float32)
        energy = np.where(valid,      subjets["energy"], 0).astype(np.float32)
        phi    = np.where(valid,      subjets["phi"],    0).astype(np.float32)
        dphi   = subjets["dphi"].astype(np.float32)
        deta   = subjets["deta"].astype(np.float32)

        cos_dphi = np.where(valid, np.cos(dphi), 1).astype(np.float32)
        sin_dphi = np.where(valid, np.sin(dphi), 0).astype(np.float32)

        new_dtype = np.dtype([
            ("pt",               np.float32),
            ("log_pt",           np.float32),
            ("log_pt_ptjet",     np.float32),
            ("mass",             np.float32),
            ("log_mass",         np.float32),
            ("log_mass_massjet", np.float32),
            ("eta",              np.float32),
            ("energy",           np.float32),
            ("phi",              np.float32),
            ("valid",            bool),
            ("dphi",             np.float32),
            ("cos_dphi",         np.float32),
            ("sin_dphi",         np.float32),
            ("deta",             np.float32),
        ])
        out = np.empty(subjets.shape, dtype=new_dtype)
        out["pt"]               = pt
        out["log_pt"]           = log_pt
        out["log_pt_ptjet"]     = log_pt_ptjet
        out["mass"]             = mass
        out["log_mass"]         = log_mass
        out["log_mass_massjet"] = log_mass_massjet
        out["eta"]              = eta
        out["energy"]           = energy
        out["phi"]              = phi
        out["valid"]            = valid
        out["dphi"]             = dphi
        out["cos_dphi"]         = cos_dphi
        out["sin_dphi"]         = sin_dphi
        out["deta"]             = deta
        return out

    def _add_subjet_fields_to_jets(self, jets: np.ndarray, subjets: np.ndarray) -> np.ndarray:
        """Append leading/subleading subjet fields to the jets structured array."""
        subjet_vars = ["pt", "log_pt", "log_pt_ptjet", "mass", "log_mass", "log_mass_massjet", "eta", "energy", "phi", "cos_dphi", "sin_dphi", "deta", "dphi"]
        leads = ["leading", "subleading"]

        subjet_field_names = {f"{lead}_subjet_{var}" for lead in leads for var in subjet_vars}
        existing = {f for f, *_ in jets.dtype.descr}
        new_fields = [
            (f"{lead}_subjet_{var}", np.float32)
            for lead in leads for var in subjet_vars
            if f"{lead}_subjet_{var}" not in existing
        ]
        new_dtype = np.dtype(jets.dtype.descr + new_fields)
        out = np.empty(jets.shape, dtype=new_dtype)
        for name in jets.dtype.names:
            if name not in subjet_field_names:
                out[name] = jets[name]
        for i, lead in enumerate(leads):
            for var in subjet_vars:
                out[f"{lead}_subjet_{var}"] = subjets[var][:, i]
        return out

    def _add_truth_total_fields_to_jets(self, jets: np.ndarray) -> np.ndarray:
        """Append flavour-dependent TruthTotalMass/TruthTotalPt fields to the jets.

        For hadronic tau-tau jets (R10TruthLabel_R22v1 in {16: HtautauHad,
        19: ZtautauHad}) the value is taken from the tau truth quantity
        (TauTruthTotal{Mass,Pt}); for all other jets (QCD in this workflow) it is
        taken from the large-R truth jet quantity (R10WZTruthLabel_R22v1_TruthJet{Mass,Pt}).
        """
        existing = {f for f, *_ in jets.dtype.descr}
        new_fields = [
            (f, np.float32) for f in self._TRUTH_TOTAL_FIELDS if f not in existing
        ]
        if not new_fields:
            return jets

        # Source fields may be absent when only a reduced jets variable set is
        # streamed (e.g. resampling's estimate_available_jets, which just counts
        # jets). Skip then — the full variable list is loaded for the actual
        # merge/write where these outputs are needed.
        required = {"R10TruthLabel_R22v1"}
        for tau_src, other_src in self._TRUTH_TOTAL_FIELDS.values():
            required.update((tau_src, other_src))
        if not required.issubset(existing):
            return jets

        use_tau_truth = np.isin(jets["R10TruthLabel_R22v1"], self._TAU_TRUTH_LABELS)

        new_dtype = np.dtype(jets.dtype.descr + new_fields)
        out = np.empty(jets.shape, dtype=new_dtype)
        for name in jets.dtype.names:
            out[name] = jets[name]
        for field_name, (tau_src, other_src) in self._TRUTH_TOTAL_FIELDS.items():
            out[field_name] = np.where(
                use_tau_truth, jets[tau_src], jets[other_src]
            ).astype(np.float32)
        return out

    def _add_is_two_subjet_to_jets(self, jets: np.ndarray) -> np.ndarray:
        """Append the binary is_two_subjet flag to the jets structured array.

        is_two_subjet = 1 when n_TruthSubjets > 1, else 0. Jets with
        n_TruthSubjets == 0 are dropped upstream by a config cut, so they never
        reach training/output; the flag value for them (0) is irrelevant.
        """
        existing = {f for f, *_ in jets.dtype.descr}
        new_fields = [
            (f, np.int8) for f in self._DERIVED_SCALAR_FIELDS if f not in existing
        ]
        if not new_fields:
            return jets

        # Skip if the source field isn't in this (possibly reduced) batch — see
        # note in _add_truth_total_fields_to_jets.
        if not all(src in existing for src in self._DERIVED_SCALAR_FIELDS.values()):
            return jets

        new_dtype = np.dtype(jets.dtype.descr + new_fields)
        out = np.empty(jets.shape, dtype=new_dtype)
        for name in jets.dtype.names:
            out[name] = jets[name]
        for field_name, src in self._DERIVED_SCALAR_FIELDS.items():
            out[field_name] = (jets[src] > 1).astype(np.int8)
        return out
