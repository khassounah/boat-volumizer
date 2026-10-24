#!/usr/bin/env bash
# Usage: volumize.sh <step-file-or-folder>
# A STEP file is volumized on its own. A folder is scanned for every .step/.stp
# file in it (not subfolders), and each one is volumized.

if [[ $# -ne 1 ]]; then
  echo "usage: $(basename "$0") <step-file-or-folder>" >&2
  exit 1
fi

target=$1
script_dir=$(cd "$(dirname "$0")" && pwd)
python=${HOME}/venv/cad/bin/python
py=${script_dir}/step_envelope_volume.py

run_one() {
  local f=$1
  echo "== $f"
  "$python" "$py" -q "$f" --plot "${f%.*}.png" 2>/dev/null || echo "ERROR: $f"
}

if [[ -d $target ]]; then
  shopt -s nullglob nocaseglob
  files=("$target"/*.step "$target"/*.stp)
  shopt -u nullglob nocaseglob
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "no STEP files in $target" >&2
    exit 1
  fi
  for f in "${files[@]}"; do
    run_one "$f"
  done
elif [[ -f $target ]]; then
  run_one "$target"
else
  echo "error: not a file or directory: $target" >&2
  exit 1
fi
