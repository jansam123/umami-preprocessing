#!/bin/bash
# Condor wrapper: build a QCD physical test dataset from a UPP *component*
# config via the preprocess CLI (high-stat, natural JZ composition).
#
# Capture the cfg arg and clear $@ BEFORE sourcing conda's activate: otherwise
# `source .../bin/activate` inherits this script's $1 (the cfg path) and tries
# to activate it as an env, printing a spurious "Not a conda environment" error.
CFG="$1"
set --

source /project/atlas/users/sjankovy/conda/bin/activate
conda activate upp

cd /project/atlas/users/sjankovy/UPP/umami-preprocessing

set -e
set -o pipefail

echo "==== job @ $(date) on $(hostname) ===="
echo "config: $CFG"
echo "python: $(which python)"

preprocess --config "$CFG" --split test --prep --resample --merge --skip-sample-check

echo "==== done @ $(date) ===="
