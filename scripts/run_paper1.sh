#!/usr/bin/env bash
# New production runs for Paper 1 (FK clusters in the 2D +-J model).
#
# Run from the repository root, inside the virtual environment:
#
#     nohup bash scripts/run_paper1.sh > results/paper1_runs.log 2>&1 &
#     tail -f results/paper1_runs.log
#
# Every job uses all cores, so jobs run one after another.  A job whose output
# file already exists is skipped, and an interrupted job resumes from its
# per-temperature checkpoints, so the script can simply be started again.
#
# Order: cheapest and most informative first.
#   1. p = 0.25, L = 16..64   exact anchor: T_1 = 2/ln 3 = 1.8205 on the Nishimori line
#   2. L = 64 at p = 0.02, 0.10, 0.15, 0.20
#   3. hot/cold-start pairs on identical disorder (equilibration floor)
#   4. L = 96 at p = 0.02, 0.05, 0.08
#   5. L = 128 at p = 0.02, 0.05, 0.08
#
# Settings follow the existing results/prod files (20 temperatures on the same
# grid per p, 2000 measured + 500 thermalisation sweeps, 20 bins, seed 1),
# except the number of disorder realizations at L >= 96, set below.

set -euo pipefail

WORKERS="${WORKERS:-$(nproc)}"
R="${R:-10000}"        # realizations for L <= 64
R96="${R96:-3000}"     # realizations for L = 96
R128="${R128:-2000}"   # realizations for L = 128
REQ="${REQ:-500}"      # realizations for hot/cold pairs
PY="${PY:-python}"

mkdir -p results/prod results/eq

window() {  # temperature window used for each p in results/prod
  case "$1" in
    0.02) echo "1.80 2.45" ;;
    0.05) echo "1.65 2.32" ;;
    0.08) echo "1.60 2.28" ;;
    0.10) echo "1.58 2.25" ;;
    0.15) echo "1.53 2.21" ;;
    0.20) echo "1.50 2.18" ;;
    0.25) echo "1.48 2.15" ;;
    *) echo "no window defined for p=$1" >&2; exit 1 ;;
  esac
}

run() {  # run L p realizations start seed out
  local L="$1" p="$2" n="$3" start="$4" seed="$5" out="$6"
  if [[ -f "$out" ]]; then
    echo "[skip] $out exists"
    return
  fi
  read -r tmin tmax <<< "$(window "$p")"
  echo "[$(date '+%F %T')] start L=$L p=$p realizations=$n start=$start -> $out"
  "$PY" scripts/run_sweep.py --L "$L" --p "$p" --t-min "$tmin" --t-max "$tmax" --n-temps 20 \
      --realizations "$n" --sweeps 2000 --therm 500 --bins 20 --seed "$seed" \
      --start "$start" --workers "$WORKERS" --save-realizations --out "$out"
  echo "[$(date '+%F %T')] done  $out"
}

# 1. Nishimori-line anchor
for L in 16 24 32 48 64; do run "$L" 0.25 "$R" hot 1 "results/prod/L${L}_p0.25.npz"; done

# 2. L = 64 where it is missing
for p in 0.20 0.15 0.10 0.02; do run 64 "$p" "$R" hot 1 "results/prod/L64_p${p}.npz"; done

# 3. hot/cold pairs on the same disorder (seed 2), full temperature grid
for p in 0.05 0.08 0.20; do
  for start in hot cold; do run 64 "$p" "$REQ" "$start" 2 "results/eq/L64_p${p}_grid_${start}.npz"; done
done

# 4. L = 96
for p in 0.08 0.05 0.02; do run 96 "$p" "$R96" hot 1 "results/prod/L96_p${p}.npz"; done

# 5. L = 128, with one hot/cold pair at the hardest point
for start in hot cold; do run 128 0.05 200 "$start" 2 "results/eq/L128_p0.05_grid_${start}.npz"; done
for p in 0.08 0.05 0.02; do run 128 "$p" "$R128" hot 1 "results/prod/L128_p${p}.npz"; done

echo "[$(date '+%F %T')] all runs finished"
