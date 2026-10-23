"""
bench — the one evaluation harness every published number comes from (D-08, EDA-13).

    python -m bench.run              # score every feature set, write results/
    python -m bench.run --dry-run    # score and print only

Pieces:
  - spec.py   : the single decoder spec and feature sets (D-12)
  - splits.py : frozen session list and calibration→test pairs (splits_v0.json)
  - plan.py   : which rows train and which rows test, for every score
  - cfd.py    : calibration-free days from per-pair kept ratios
  - run.py    : load data, execute the plan, write the scoreboard
"""
