"""Block until the qdeep queue log shows it exited on its STOP file (after forward_inner)."""
import time
from pathlib import Path
log = Path(__file__).resolve().parent / "qdeep.queue.log"
while not (log.exists() and "STOP file present" in log.read_text(encoding="utf-8")):
    time.sleep(60)
print("qdeep stopped after forward_inner; starting full")
