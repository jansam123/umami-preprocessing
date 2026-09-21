#!/usr/bin/env python
"""Create a merged Ztt physical testing dataset with subjet transforms and kinematic cuts.

Reads input HDF5 files specified in the config, applies SubjetTransform (log-pt,
log-mass, cos/sin-dphi for truth_subjets; leading/subleading subjet fields added
to jets), applies kinematic cuts, and writes all jets into a single output file.

No resampling or reweighting is performed.

Usage:
    python create_test_dataset.py [--config PATH]

Default config: upp/configs/ztt_test.yaml
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import yaml
from ftag import Cuts
from ftag.hdf5 import H5Reader, H5Writer
from yamlinclude import YamlIncludeConstructor

from upp.classes.variable_config import VariableConfig
from upp.utils.subjet_transforms import SubjetTransform


def load_config(config_path: Path) -> dict:
    YamlIncludeConstructor.add_to_loader_class(
        loader_class=yaml.SafeLoader,
        base_dir=str(config_path.parent),
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "configs/ztt_test.yaml",
        help="Path to the YAML config file (default: upp/configs/ztt_test.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    jets_name = cfg["jets_name"]

    var_cfg = VariableConfig(cfg["variables"], jets_name=jets_name)
    variables = var_cfg.combined()  # dict[group -> list[str]]

    transform = SubjetTransform(**cfg.get("transform", {}))
    cuts = Cuts.from_list(cfg["cuts"])
    input_files = sorted(
        Path(f) for pattern in cfg["input_files"] for f in glob.glob(pattern)
    )

    reader = H5Reader(
        fname=input_files,
        batch_size=cfg["batch_size"],
        shuffle=False,
        transform=transform,
        jets_name=jets_name,
    )

    num_jets = reader.num_jets  # pre-cut total, used only for shape inference
    dtypes = reader.dtypes(variables)
    dtypes = {name: transform.output_dtype(name, dtype) for name, dtype in dtypes.items()}
    shapes = reader.shapes(num_jets, list(variables.keys()))

    output = Path(cfg["output_file"])
    output.parent.mkdir(parents=True, exist_ok=True)

    # Dynamic mode (num_jets omitted): datasets start at size 0 and grow with
    # each write call. This gives the correct output size after cuts without
    # needing to pre-estimate the post-cut jet count.
    writer = H5Writer(
        dst=output,
        dtypes=dtypes,
        shapes=shapes,
        jets_name=jets_name,
    )

    written = 0
    for batch in reader.stream(variables, num_jets, cuts=cuts):
        writer.write(batch)
        written += len(batch[jets_name])

    writer.close()
    print(f"Done — wrote {written:,} jets to {output}")


if __name__ == "__main__":
    main()
