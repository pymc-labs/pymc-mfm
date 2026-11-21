"""One agent script, run as its own process by ``test_duplicate_work.py``.

Usage: ``analyze_once.py BASE_URL [SEED]``. Starts one backtest analysis, prints its id, then
waits at most one short poll. The analysis never ends because the stub holds its worker.
"""

import sys

from pymc_mktfm.client import Client

base_url = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
client = Client(base_url, "test-key", max_retries=0)
analysis = client.analyze(
    "ds_1", context=("2024-07-01", "2024-07-15"), query_end="2024-08-05", seed=seed
)
print(analysis.id, flush=True)
try:
    analysis.wait(timeout=0, poll=0.01, announce=False)
except TimeoutError:
    print("still running", flush=True)
