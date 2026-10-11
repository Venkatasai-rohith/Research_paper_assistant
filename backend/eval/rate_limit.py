
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Callable, TypeVar


def env_float(name: str, default: float) -> float:
    """Read a floating-point setting from an environment variable."""
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"Environment variable {name} must be a valid number; "
            f"received {value!r}"
        )


def env_int(name: str, default: int) -> int:
    """Read an integer setting from an environment variable."""
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"Environment variable {name} must be a valid integer; "
            f"received {value!r}"
        )


T = TypeVar("T")


@dataclass
class EvalRunConfig:
    """Configuration for evaluation runs."""

    request_delay_sec: float = 3.0
    max_retries: int = 6
    backoff_base_sec: float = 45.0
    case_delay_sec: float = 10.0
    max_cases: int | None = None
    skip_judge: bool = False
    serial_summaries: bool = True


_config = EvalRunConfig()
_last_request_time = 0.0


def configure_eval(
    config: EvalRunConfig | None = None,
) -> EvalRunConfig:
    """Configure evaluation settings from arguments and environment."""

    global _config

    if config is None:
        config = EvalRunConfig(
            request_delay_sec=env_float(
                "EVAL_REQUEST_DELAY_SEC", 3.0
            ),
            max_retries=env_int(
                "EVAL_MAX_RETRIES", 6
            ),
            backoff_base_sec=env_float(
                "EVAL_BACKOFF_BASE_SEC", 45.0
            ),
            case_delay_sec=env_float(
                "EVAL_CASE_DELAY_SEC", 10.0
            ),
            max_cases=(
                env_int("EVAL_MAX_CASES", 0) or None
            ),
            skip_judge=os.getenv(
                "EVAL_SKIP_JUDGE", "false"
            ).strip().lower() in {"1", "true", "yes"},
            serial_summaries=os.getenv(
                "EVAL_SERIAL_SUMMARIES", "true"
            ).strip().lower() in {"1", "true", "yes"},
        )

    _config = config
    apply_summary_worker_limit()

    print(
        "[Eval config] "
        f"request_delay={_config.request_delay_sec}s, "
        f"max_retries={_config.max_retries}, "
        f"backoff_base={_config.backoff_base_sec}s, "
        f"case_delay={_config.case_delay_sec}s, "
        f"max_cases={_config.max_cases}, "
        f"skip_judge={_config.skip_judge}, "
        f"serial_summaries={_config.serial_summaries}"
    )

    return _config


def get_config() -> EvalRunConfig:
    """Return the current evaluation configuration."""
    return _config


def apply_summary_worker_limit() -> None:
    """Limit summary map-reduce workers when serial execution is enabled."""

    if _config.serial_summaries:
        os.environ["SUMMARY_MAP_REDUCE_MAX_WORKERS"] = "1"


def throttle() -> None:
    """Ensure a minimum delay between consecutive requests."""

    global _last_request_time

    delay = _config.request_delay_sec

    if delay <= 0:
        _last_request_time = time.monotonic()
        return

    now = time.monotonic()
    elapsed = now - _last_request_time
    remaining = delay - elapsed

    if remaining > 0:
        time.sleep(remaining)

    _last_request_time = time.monotonic()



def pause_after_case(label: str | None = None) -> None:
    """Pause between evaluation cases to reduce API request pressure."""

    if _config.case_delay_sec > 0:
        if label:
            print(
                f"[{label}] Waiting {_config.case_delay_sec}s "
                "before the next case..."
            )
        time.sleep(_config.case_delay_sec)



def limit_cases(cases: list[T]) -> list[T]:
    """Apply the optional maximum number of cases to evaluate."""

    if _config.max_cases is None:
        return cases

    limited_cases = cases[:_config.max_cases]

    if len(limited_cases) < len(cases):
        print(
            f"[Eval config] Limiting evaluation to "
            f"{len(limited_cases)} of {len(cases)} cases."
        )

    return limited_cases


def error_message(exc: BaseException) -> str:
    """Return a normalized exception message."""

    return str(exc).lower()


def is_rate_limit_error(exc: BaseException | str) -> bool:
    """Detect common API rate-limit errors."""

    msg = (
        error_message(exc)
        if isinstance(exc, BaseException)
        else exc.lower()
    )

    indicators = (
        "429",
        "rate_limit",
        "rate limit",
        "too many requests",
        "resource exhausted",
    )

    return any(indicator in msg for indicator in indicators)


def is_daily_quota_error(exc: BaseException | str) -> bool:
    """Detect daily token/request quota exhaustion."""

    msg = (
        error_message(exc)
        if isinstance(exc, BaseException)
        else exc.lower()
    )

    if not is_rate_limit_error(msg):
        return False

    daily_indicators = (
        "tokens per day",
        "(tpd)",
        " tpd",
        "daily quota",
        "daily limit",
    )

    return any(
        indicator in msg
        for indicator in daily_indicators
    )


def parse_retry_after(exc: BaseException) -> float | None:
    """Best-effort extraction of a retry delay from an exception."""

    msg = str(exc)

    patterns = (
        r"retry(?:\s+|-)?after['\"]?\s*[:=]?\s*(\d+(?:\.\d+)?)",
        r"try again in (\d+(?:\.\d+)?)\s*s",
    )

    for pattern in patterns:
        match = re.search(pattern, msg, re.IGNORECASE)

        if match:
            return float(match.group(1))

    return None


def retry_wait_seconds(attempt: int) -> float:
    """Calculate a linearly increasing delay between retries."""

    return _config.backoff_base_sec * (attempt + 1)


def call_with_retry(
    fn: Callable[[], T],
    *,
    label: str = "Evaluation call",
    stop_on_daily_quota: bool = True,
) -> tuple[T | None, str | None]:
    """
    Execute a callable with rate-limit retries.

    Returns:
        (result, None) on success.
        (None, "daily_quota") when a daily quota is exhausted and
        stop_on_daily_quota is True.
        (None, None) when the operation fails without a result.
    """

    for attempt in range(_config.max_retries + 1):
        throttle()

        try:
            result = fn()
            return result, None

        except Exception as exc:
            msg = str(exc)

            if is_daily_quota_error(exc):
                print(
                    f"[{label}] Daily API quota exhausted: {msg}"
                )

                if stop_on_daily_quota:
                    print(
                        f"[{label}] Stopping this evaluation "
                        "to avoid unnecessary API calls."
                    )
                    return None, "daily_quota"

                return None, None

            if is_rate_limit_error(exc):
                if attempt >= _config.max_retries:
                    print(
                        f"[{label}] Rate limit persisted after "
                        f"{_config.max_retries} retries: {msg}"
                    )
                    return None, None

                wait_seconds = (
                    parse_retry_after(exc)
                    or retry_wait_seconds(attempt)
                )

                wait_seconds = max(0.0, wait_seconds)

                print(
                    f"[{label}] Rate limit encountered. "
                    f"Retry {attempt + 1}/{_config.max_retries} "
                    f"in {wait_seconds:.1f}s."
                )

                time.sleep(wait_seconds)
                continue

            print(f"[{label}] Failed: {msg}")
            return None, None

    return None, None


def invoke_llm(
    llm,
    prompt: str,
    *,
    label: str = "LLM invoke",
    stop_on_daily_quota: bool = True,
):
    """Invoke an LLM with retry and daily-quota handling."""

    result, stop = call_with_retry(
        lambda: llm.invoke(prompt),
        label=label,
        stop_on_daily_quota=stop_on_daily_quota,
    )

    return result, stop
