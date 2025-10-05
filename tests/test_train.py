import json
import os
import subprocess
import sys
from pathlib import Path

def test_training_script_runs(tmp_path):
    # Copy dataset into temporary location
    data_src = Path('data/raw/house_prices.csv')
    data_dst = tmp_path / 'house_prices.csv'
    data_dst.write_bytes(data_src.read_bytes())

    # Run training into temp output dir
    outdir = tmp_path / 'models'
    cmd = [sys.executable, 'src/train.py', '--data', str(data_dst), '--outdir', str(outdir)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f'Non-zero exit: {r.stderr}\\n{r.stdout}'

    metrics_file = outdir / 'metrics.json'
    assert metrics_file.exists(), 'metrics.json was not created'

    with open(metrics_file) as f:
        metrics = json.load(f)
    # Basic sanity checks
    assert 0.0 <= metrics['r2'] <= 1.0
    assert metrics['rmse'] > 0
