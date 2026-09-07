#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the target Conda environment first." >&2
  exit 1
fi

repo_dir="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
timm_version="$(python -c 'import timm; print(timm.__version__)')"

if [[ "$timm_version" != "0.6.13" ]]; then
  echo "Expected timm 0.6.13, found $timm_version." >&2
  exit 1
fi

timm_models_dir="$(python -c 'import pathlib, timm; print(pathlib.Path(timm.__file__).parent / "models")')"
original_resnet="$timm_models_dir/resnet.py.timm-0.6.13.orig"

if [[ ! -f "$original_resnet" ]]; then
  cp "$timm_models_dir/resnet.py" "$original_resnet"
fi

cp "$repo_dir/input_masking/resnet.py" "$timm_models_dir/resnet.py"
cp "$repo_dir/input_masking/sal_layers.py" "$timm_models_dir/sal_layers.py"

python -m py_compile "$timm_models_dir/resnet.py" "$timm_models_dir/sal_layers.py"
echo "Installed HU-MCD masked ResNet into $timm_models_dir"
