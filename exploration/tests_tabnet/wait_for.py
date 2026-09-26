"""Block until a run_queue log says 'queue done' (python sleep; used as the first queue step)."""
import sys, time
from pathlib import Path
log = Path(sys.argv[1])
while not (log.exists() and "queue done" in log.read_text(encoding="utf-8")):
    time.sleep(60)
print(f"{log} done")
