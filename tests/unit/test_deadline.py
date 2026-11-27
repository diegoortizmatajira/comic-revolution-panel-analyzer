import time

import pytest
from app.core.deadline import check_deadline, time_limit
from app.core.errors import AnalysisTimeoutError


def test_no_limit_never_raises() -> None:
    check_deadline()


def test_raises_once_the_limit_has_passed() -> None:
    with time_limit(0.01):
        check_deadline()
        time.sleep(0.03)
        with pytest.raises(AnalysisTimeoutError):
            check_deadline()


def test_limit_ends_with_the_block() -> None:
    with time_limit(0.01):
        pass
    time.sleep(0.03)
    check_deadline()
