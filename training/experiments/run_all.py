"""Execute the experiment notebooks in order, logging each one.

    python run_all.py              # everything
    python run_all.py 03 04        # just those

Each notebook is executed in place, so the committed .ipynb carries its outputs.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["01_partition_and_data_audit", "02_scorer_variants",
         "03_clean_partition_table4", "04_reward_sensitivity",
         "05_independent_traffic", "06_stochastic_action_model",
         "07_proxy_reward", "08_data_sufficiency"]
TIMEOUT = 14400


def run(stem: str) -> bool:
    nb = HERE / f"{stem}.ipynb"
    log = HERE / f"logs_{stem[:2]}.txt"
    print(f"\n=== {stem} ===", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--inplace",
             f"--ExecutePreprocessor.timeout={TIMEOUT}", str(nb)],
            stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    mins = (time.time() - t0) / 60
    ok = p.returncode == 0
    print(f"{'OK  ' if ok else 'FAIL'} {stem}  ({mins:.1f} min)  -> {log.name}", flush=True)
    if not ok:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        print("\n".join(tail), flush=True)
    return ok


if __name__ == "__main__":
    want = sys.argv[1:]
    targets = [s for s in ORDER if not want or any(s.startswith(w) for w in want)]
    failed = [s for s in targets if not run(s)]
    print("\n" + ("all notebooks completed" if not failed
                  else f"FAILED: {', '.join(failed)}"))
    sys.exit(1 if failed else 0)
