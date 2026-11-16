"""External watchdog for chain elements that must not run unwatched (power amplifiers).

The watchdog is a separate process, started before the HIL session and independent
of it, that watches the hardware (e.g. amplifier temperature and current) and stops
the radar itself on an unsafe reading. It reports through a JSON status file,
rewritten at least every second:

    {"state": "armed", "updated_unix": 1763320000.1, "expires_unix": 1763321800.0,
     "powered": true, "reason": null, "pid": 1234}

state is "armed" while it is watching and will stop the radar on a trip; anything
else ("starting", "tripped", "expired", "stopped") means no captures. The harness
checks the status before every capture (which must end before expires_unix), and
while a capture runs it stops the radar if the status goes stale or leaves "armed".
"""
import json
import signal
import threading
import time
from pathlib import Path

DEFAULT_MAX_AGE_S = 3.0
POLL_S = 0.5
SIGINT_GRACE_S = 5.0


class WatchdogError(RuntimeError):
    pass


def read_status(spec):
    path = Path(spec["status_file"]).expanduser()
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise WatchdogError(f"watchdog status {path}: {e}")


def problem(spec, min_remaining_s=0.0, now=None):
    """Why captures may not run now, or None."""
    now = time.time() if now is None else now
    try:
        st = read_status(spec)
    except WatchdogError as e:
        return str(e)
    age = now - float(st.get("updated_unix", 0))
    if age > float(spec.get("max_age_s", DEFAULT_MAX_AGE_S)):
        return f"watchdog status is {age:.1f} s old"
    if st.get("state") != "armed":
        return f"watchdog is {st.get('state')!r}" + (f": {st['reason']}" if st.get("reason") else "")
    if not st.get("powered"):
        return "watchdog reports the amplifier unpowered"
    remaining = float(st.get("expires_unix", 0)) - now
    if remaining < min_remaining_s:
        return f"watchdog expires in {remaining:.0f} s, capture needs up to {min_remaining_s:.0f} s"
    return None


def check(specs, min_remaining_s=0.0):
    """Raise WatchdogError unless every watchdog in specs [(element id, spec)] is armed."""
    for el_id, spec in specs:
        why = problem(spec, min_remaining_s)
        if why:
            raise WatchdogError(f"{el_id}: {why}")


class Monitor:
    """Stops ``proc`` (SIGINT, then SIGKILL) if any watchdog stops being armed while it runs."""

    def __init__(self, specs, proc):
        self.specs, self.proc = specs, proc
        self.failure = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(POLL_S) and self.proc.poll() is None:
            for el_id, spec in self.specs:
                why = problem(spec)
                if why:
                    self.failure = f"{el_id}: {why}"
                    self._stop_radar()
                    return

    def _stop_radar(self):
        try:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=SIGINT_GRACE_S)
        except Exception:
            self.proc.kill()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)
