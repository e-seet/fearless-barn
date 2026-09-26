#!/usr/bin/env python3
"""Run a list of shell commands one after another, safely stoppable.

Why this exists (2026-09-26): twice this build a queue misbehaved.
* A `until grep ...; do sleep; done` waiter never fired and idled the GPU ~1.5 h.
* Killing a training process let its bash launcher advance to the next stage.

This driver:
* writes its own PID to `<name>.pid` and the running child's PID to `<name>.child.pid`;
* checks for a `<name>.STOP` file before starting each command, and exits cleanly if present;
* logs start/end/exit code of every step to `<name>.queue.log` with timestamps.

Stop cleanly: create the STOP file (the current step finishes, nothing new starts).
Stop now: create the STOP file, then kill the child PID; the driver sees the STOP
file and does not advance.

usage: python run_queue.py <name> <commands_file>   (one shell command per line; # comments ok)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    name, cmd_file = sys.argv[1], Path(sys.argv[2])
    base = Path(name)
    stop, pidf, childf, log = (base.with_suffix(".STOP"), base.with_suffix(".pid"),
                               base.with_suffix(".child.pid"), base.with_suffix(".queue.log"))
    cmds = [l.strip() for l in cmd_file.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")]
    pidf.write_text(str(os.getpid()))
    stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log, "a", encoding="utf-8") as L:
        L.write(f"{stamp()} queue start: {len(cmds)} steps, driver pid {os.getpid()}\n"); L.flush()
        for i, c in enumerate(cmds, 1):
            if stop.exists():
                L.write(f"{stamp()} STOP file present before step {i}; exiting\n"); L.flush()
                return 0
            L.write(f"{stamp()} step {i}/{len(cmds)} START: {c}\n"); L.flush()
            p = subprocess.Popen(c, shell=True)
            childf.write_text(str(p.pid))
            rc = p.wait()
            L.write(f"{stamp()} step {i}/{len(cmds)} END rc={rc}\n"); L.flush()
            if rc != 0:
                L.write(f"{stamp()} non-zero exit; stopping the queue\n"); L.flush()
                return rc
        L.write(f"{stamp()} queue done\n")
    childf.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
