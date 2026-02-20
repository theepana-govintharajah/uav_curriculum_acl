\
@echo off
setlocal
call .venv\Scripts\activate
python scripts/train.py --schedule resource_aware --seed 0
python scripts/train.py --schedule none --seed 0
python scripts/train.py --schedule linear --seed 0
python scripts/train.py --schedule random --seed 0
python scripts/eval.py --runs_dir runs
python scripts/plot.py --runs_dir runs
endlocal
