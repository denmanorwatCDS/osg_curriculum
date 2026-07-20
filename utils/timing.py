"""
Lightweight, dependency-free latency profiler for the OSG pipeline.

Usage:
    from utils.timing import PROFILER
    PROFILER.start_episode(episode_id, scene=..., goal=...)
    with PROFILER.section("observe"):
        ...
    PROFILER.record("obs_to_action", seconds)
    PROFILER.end_episode()   # flushes one JSON line per episode

Output: one JSON object per completed episode appended to
$OSG_LATENCY_LOG (default: logs/latency.jsonl). Disable with OSG_PROFILE=0.

Design notes:
- Per-episode we accumulate (total_seconds, count) per named section, so global
  means/percentiles can be recomputed later from the raw JSONL without re-running
  the (GPT-token-costly) evaluation.
- Every public method is exception-safe: a profiling failure must never break a run.
"""

import os
import json
import time
from contextlib import contextmanager


class Profiler:
    def __init__(self):
        self.enabled = os.environ.get("OSG_PROFILE", "1") != "0"
        self.path = os.environ.get("OSG_LATENCY_LOG", "logs/latency.jsonl")
        self.episode = None
        self.scene = None
        self.goal = None
        self._ep_start = None
        self._agg = {}  # section -> [total_seconds, count]

    def start_episode(self, episode_id, scene=None, goal=None):
        self.episode = episode_id
        self.scene = scene
        self.goal = goal
        self._agg = {}
        self._ep_start = time.perf_counter()

    def record(self, section, seconds):
        if not self.enabled:
            return
        slot = self._agg.get(section)
        if slot is None:
            self._agg[section] = [seconds, 1]
        else:
            slot[0] += seconds
            slot[1] += 1

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - t0)

    def end_episode(self):
        if not self.enabled or self.episode is None:
            return
        try:
            total = time.perf_counter() - self._ep_start if self._ep_start else 0.0
            record = {
                "episode_id": self.episode,
                "scene": self.scene,
                "goal": self.goal,
                "episode_total_s": total,
                "sections": {
                    name: {
                        "total_s": tot,
                        "count": cnt,
                        "mean_s": (tot / cnt) if cnt else 0.0,
                    }
                    for name, (tot, cnt) in self._agg.items()
                },
            }
            if self.path:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(record) + "\n")
        except Exception as exc:  # never let profiling break the eval
            print("[PROFILER] failed to write latency record:", repr(exc))
        finally:
            self.episode = None
            self._agg = {}


PROFILER = Profiler()


# --- OpenAI client proxy that times chat.completions.create as "llm_api" ------

class _TimedCompletions:
    def __init__(self, inner):
        self._inner = inner

    def create(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self._inner.create(*args, **kwargs)
        finally:
            PROFILER.record("llm_api", time.perf_counter() - t0)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _TimedChat:
    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "completions", _TimedCompletions(inner.completions))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _TimedOpenAI:
    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "chat", _TimedChat(inner.chat))

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        # Pass attribute writes (e.g. api_key) through to the real module.
        setattr(self._inner, name, value)


def timed_openai(openai_module):
    """Wrap the openai module so .chat.completions.create is timed. On any
    structural mismatch, raises and the caller should fall back to the raw module."""
    return _TimedOpenAI(openai_module)
