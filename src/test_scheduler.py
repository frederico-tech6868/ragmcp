import time

from scheduler import start_scheduler


def wait_until(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            return False
        time.sleep(0.005)
    return True


def test_runs_repeatedly():
    calls = []
    scheduler = start_scheduler(interval=0.01, run=lambda: calls.append(1))
    try:
        assert wait_until(lambda: len(calls) >= 3)
    finally:
        scheduler.stop()


def test_stop_ends_the_loop_right_away():
    calls = []
    scheduler = start_scheduler(interval=60, run=lambda: calls.append(1))
    assert wait_until(lambda: calls)
    started = time.monotonic()
    scheduler.stop(timeout=2)
    assert not scheduler.thread.is_alive()
    assert time.monotonic() - started < 1   # didn't sit out the 60 s interval
    count = len(calls)
    time.sleep(0.05)
    assert len(calls) == count == 1


def test_exception_does_not_kill_the_loop():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("boom")

    scheduler = start_scheduler(interval=0.01, run=flaky)
    try:
        assert wait_until(lambda: len(calls) >= 3)
        assert scheduler.thread.is_alive()
    finally:
        scheduler.stop()
