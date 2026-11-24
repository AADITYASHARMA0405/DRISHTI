from drishti.utils import KeepAwake, on_battery, wait_for_mains


def test_wait_for_mains_pauses_until_plugged_in():
    states = iter([True, True, True, False])          # on battery for three checks, then mains
    msgs = []
    waited = wait_for_mains(check=lambda: next(states), poll_s=0.01, log=msgs.append)
    assert waited > 0
    assert "paused" in msgs[0] and "resuming" in msgs[-1]


def test_wait_for_mains_is_instant_on_mains():
    assert wait_for_mains(check=lambda: False, poll_s=0.01) == 0.0


def test_power_helpers_run_here():
    assert isinstance(on_battery(), bool)
    with KeepAwake() as k:
        assert k.flags & KeepAwake.ES_DISPLAY_REQUIRED
    assert not KeepAwake(keep_display_on=False).flags & KeepAwake.ES_DISPLAY_REQUIRED
