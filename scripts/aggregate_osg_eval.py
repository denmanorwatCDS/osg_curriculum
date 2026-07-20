#!/usr/bin/env python3
"""
Aggregate an OSG ObjectNav evaluation: navigation metrics (excluding crashes)
plus per-submodule latency, from the run log(s) and the latency JSONL.

Usage:
    python scripts/aggregate_osg_eval.py \
        --logs logs/osg_eval_100.log [logs/osg_eval_100_resume.log ...] \
        --latency logs/latency.jsonl \
        [--csv logs/osg_eval_100_summary.csv]

Navigation metrics are parsed from the stdout produced by navigate_homerobot.py's
__main__ loop (RUN / EPISODE / SCENE / GOAL markers) and the per-episode Habitat
metrics dict printed at the end of each episode. An attempt with markers but no
metrics dict is counted as a crash and excluded from the metric averages.

Latency is read from the JSONL emitted by utils/timing.PROFILER (one line per
completed episode). Because raw per-section totals+counts are stored, you can
re-run this aggregation for different views without repeating the evaluation.
"""

import argparse
import ast
import json
import os
import re
import statistics
from collections import defaultdict


RUN_RE = re.compile(r"^RUN\s+(\d+)\s*/\s*(\d+)")
EPISODE_RE = re.compile(r"^EPISODE\s+(.+?)\s*$")
SCENE_RE = re.compile(r"^SCENE\s+(.+?)\s*$")
GOAL_RE = re.compile(r"^GOAL\s+(.+?)\s*$")
DICT_RE = re.compile(r"\{.*\}")


def short_scene(scene):
    """'.../00033-oPj9qMxrDEa/oPj9qMxrDEa.basis.glb' -> 'oPj9qMxrDEa'."""
    if not scene:
        return scene
    base = os.path.basename(scene.strip())
    return base.split(".")[0] if base else scene


def parse_metrics_dict(line):
    m = DICT_RE.search(line)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group(0))
    except Exception:
        return None
    if isinstance(d, dict) and "success" in d and "spl" in d:
        return d
    return None


def parse_runs(log_paths):
    """Return list of attempts: dicts with episode_id, scene, goal, metrics(or None)."""
    attempts = []
    cur = None

    def flush():
        if cur is not None:
            attempts.append(cur)

    for path in log_paths:
        if not os.path.exists(path):
            print(f"[warn] log not found: {path}")
            continue
        with open(path, "r", errors="replace") as f:
            for raw in f:
                line = raw.rstrip("\n")
                s = line.strip()
                if RUN_RE.match(s):
                    flush()
                    cur = {"episode_id": None, "scene": None, "goal": None, "metrics": None}
                    continue
                if cur is None:
                    continue
                m = EPISODE_RE.match(s)
                if m:
                    cur["episode_id"] = m.group(1)
                    continue
                m = SCENE_RE.match(s)
                if m:
                    cur["scene"] = m.group(1)
                    continue
                m = GOAL_RE.match(s)
                if m:
                    cur["goal"] = m.group(1)
                    continue
                if cur["metrics"] is None and "'success'" in s and "'spl'" in s:
                    d = parse_metrics_dict(s)
                    if d is not None:
                        cur["metrics"] = d
    flush()
    return attempts


def mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize_metrics(rows, title):
    if not rows:
        return
    succ = [float(r["success"]) for r in rows]
    spl = [float(r["spl"]) for r in rows]
    sspl = [float(r.get("soft_spl", 0.0)) for r in rows]
    dtg = [float(r["distance_to_goal"]) for r in rows]
    print(f"  {title:<24} n={len(rows):<4} "
          f"success={mean(succ)*100:6.2f}%  spl={mean(spl):.4f}  "
          f"soft_spl={mean(sspl):.4f}  dist_to_goal={mean(dtg):.4f} m")


def report_navigation(attempts):
    completed = [a for a in attempts if a["metrics"] is not None]
    crashed = [a for a in attempts if a["metrics"] is None]

    rows = []
    for a in completed:
        d = a["metrics"]
        rows.append({
            "scene": short_scene(a["scene"]),
            "goal": a["goal"],
            "success": d.get("success", 0.0),
            "spl": d.get("spl", 0.0),
            "soft_spl": d.get("soft_spl", 0.0),
            "distance_to_goal": d.get("distance_to_goal", 0.0),
        })

    print("=" * 78)
    print("NAVIGATION METRICS")
    print("=" * 78)
    print(f"Attempted: {len(attempts)}   Completed: {len(completed)}   "
          f"Crashed (excluded): {len(crashed)}")
    print()
    print("Overall (completed only, excluding crashes):")
    summarize_metrics(rows, "ALL")

    if rows:
        n = len(attempts)
        succ_sum = sum(float(r["success"]) for r in rows)
        spl_sum = sum(float(r["spl"]) for r in rows)
        print()
        print("Overall (crashes counted as failures):")
        print(f"  {'ALL':<24} n={n:<4} success={succ_sum/n*100:6.2f}%  spl={spl_sum/n:.4f}"
              if n else "")

    by_scene = defaultdict(list)
    by_goal = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)
        by_goal[r["goal"]].append(r)

    print()
    print("Per scene:")
    for scene in sorted(by_scene):
        summarize_metrics(by_scene[scene], scene)
    print()
    print("Per goal category:")
    for goal in sorted(by_goal, key=lambda g: (g is None, g)):
        summarize_metrics(by_goal[goal], str(goal))

    if crashed:
        print()
        print("Crashed / excluded attempts:")
        for a in crashed:
            print(f"  scene={short_scene(a['scene'])} episode={a['episode_id']} goal={a['goal']}")
    return rows


def report_latency(latency_path):
    if not os.path.exists(latency_path):
        print(f"\n[warn] latency log not found: {latency_path}")
        return
    records = []
    with open(latency_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    if not records:
        print(f"\n[warn] no latency records in {latency_path}")
        return

    ep_totals = [r.get("episode_total_s", 0.0) for r in records]

    # section -> total_seconds, total_calls, and per-episode totals
    sec_time = defaultdict(float)
    sec_calls = defaultdict(int)
    sec_ep_time = defaultdict(list)
    for r in records:
        for name, v in r.get("sections", {}).items():
            sec_time[name] += v.get("total_s", 0.0)
            sec_calls[name] += v.get("count", 0)
            sec_ep_time[name].append(v.get("total_s", 0.0))

    n_ep = len(records)

    def ms_calls(name):
        """(mean ms per call, total calls) for a section, or None if never recorded."""
        c = sec_calls.get(name, 0)
        if not c:
            return None
        return sec_time[name] / c * 1000.0, c

    print()
    print("=" * 78)
    print(f"LATENCY  (per-episode records: {n_ep})")
    print("=" * 78)
    print(f"episode_total (s):  mean={mean(ep_totals):8.2f}  median={pct(ep_totals,0.5):8.2f}  "
          f"p90={pct(ep_totals,0.9):8.2f}  min={min(ep_totals):8.2f}  max={max(ep_totals):8.2f}")

    # --- Decision latency: obs-ready -> action, split by step type ------------
    print()
    print("DECISION LATENCY  (per emitted action)")
    decision_rows = [
        ("obs_to_action_no_viz", "MAIN — obs->action, excl. debug visualise"),
        ("obs_to_action_all", "blended over all emitted actions"),
        ("obs_to_action_reasoning", "steps WITH high-level reasoning (high_level_loop>0)"),
        ("obs_to_action_control", "pure control steps (high_level_loop==0)"),
    ]
    for name, meaning in decision_rows:
        r = ms_calls(name)
        star = "*" if name == "obs_to_action_no_viz" else " "
        if r is None:
            print(f" {star} {name:<26} {'n/a':>10}            {meaning}")
        else:
            ms, c = r
            print(f" {star} {name:<26} {ms:8.1f} ms  (n={c:<4}) {meaning}")

    # --- Submodule latency ----------------------------------------------------
    print()
    print("SUBMODULE LATENCY  (mean per call)")
    print(f"  {'metric':<20}{'calls/ep':>10}{'ms/call':>12}   meaning")
    print("  " + "-" * 74)
    sub_rows = [
        ("high_level_loop", "BLIP+GPT+planner+scene-graph reasoning span"),
        ("llm_api", "GPT API call only"),
        ("vqa", "BLIP / VQA"),
        ("detect", "GroundingDINO detection"),
        ("perceive", "perception block (contains vqa/detect + some llm_api)"),
        ("scene_graph_update", "OSGUpdater"),
        ("planner", "Planner"),
        ("mapper_update", "obs -> map update"),
        ("controller_step", "FMM controller -> action"),
        ("set_subgoal", "controller.set_subgoal_image"),
        ("observe", "observation gathering (NOT in decision latency)"),
        ("visualise", "debug UI render (excluded from the main number)"),
    ]
    for name, meaning in sub_rows:
        r = ms_calls(name)
        if r is None:
            continue
        ms, c = r
        print(f"  {name:<20}{c/n_ep:>10.2f}{ms:>12.1f}   {meaning}")

    print()
    print("Notes:")
    print("  * obs_to_action_no_viz = obs_to_action_all - visualise (per action).")
    print("  * Reasoning-step latency still includes matplotlib debug image saves inside")
    print("    high_level_loop (visualise_objects / visualise_chosen_goal), which the")
    print("    'visualise' subtraction does not remove.")
    print("  * high_level_loop / perceive are spans that CONTAIN llm_api, vqa, detect, etc.")


def maybe_write_csv(rows, csv_path):
    if not csv_path:
        return
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "goal", "success", "spl", "soft_spl", "distance_to_goal"])
        for r in rows:
            w.writerow([r["scene"], r["goal"], r["success"], r["spl"],
                        r["soft_spl"], r["distance_to_goal"]])
    print(f"\nPer-episode CSV written: {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=["logs/osg_eval_100.log"],
                    help="run log file(s), in order")
    ap.add_argument("--latency", default="logs/latency.jsonl",
                    help="latency JSONL emitted by utils.timing.PROFILER")
    ap.add_argument("--csv", default=None, help="optional per-episode CSV output path")
    args = ap.parse_args()

    attempts = parse_runs(args.logs)
    rows = report_navigation(attempts)
    report_latency(args.latency)
    maybe_write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
