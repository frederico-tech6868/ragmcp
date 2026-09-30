"""Run a job (skill curation by default) at a fixed interval in a background thread."""

import logging
import threading

from curation import curate_skills

logging.basicConfig(level=logging.INFO)

DAY = 24 * 60 * 60


class Scheduler:
    def __init__(self, interval, run):
        self.interval = interval
        self.run = run
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.run()
            except Exception:
                logging.exception("Scheduled job failed")
            self._stop.wait(self.interval)   # returns at once when stop() is called

    def stop(self, timeout=None):
        """Stop the loop and wait for the thread (a job in progress finishes first)."""
        self._stop.set()
        self.thread.join(timeout)


def start_scheduler(interval=DAY, run=curate_skills):
    """Run `run()` now and then every `interval` seconds; returns a Scheduler with stop()."""
    scheduler = Scheduler(interval, run)
    scheduler.thread.start()
    logging.info("Scheduler started (every %s s)", interval)
    return scheduler


if __name__ == "__main__":
    scheduler = start_scheduler()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        scheduler.stop()
        logging.info("Scheduler shutdown")
