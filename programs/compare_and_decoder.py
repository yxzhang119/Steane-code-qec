"""Plot the seven-scheme comparison and measure pure QEC decoder time.

Run this file with no arguments for the retained logical-error plot and a small
fresh decoder benchmark. Only numpy, stim and the standard library are needed.
The seven neighbouring scheme files supply their own circuits and controllers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import csv
from datetime import datetime
from functools import wraps
import hashlib
from html import escape
import importlib.util
import math
from pathlib import Path
import platform
import random
from statistics import mean, median, stdev
import sys
from time import perf_counter_ns

import numpy as np
import stim

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent if ROOT.name == "programs" and (ROOT.parent / "requirements.txt").is_file() else ROOT
SCHEMES = {
    "steane_1": ("steane_one_SteaneCode", "Steane-1"),
    "steane_2": ("steane_two_SteaneCode", "Steane-2"),
    "shor_1": ("shor_one_SteaneCode", "Shor-1"),
    "shor_3": ("shor_adaptive_three", "Shor-3"),
    "shor_4": ("shor_confirm_four", "Shor-4"),
    "sequential_flag": ("sequential_flag", "Seq-flag"),
    "parallel": ("parallel", "Parallel"),
}
CATEGORIES = ("control", "decoder", "stim", "frame", "scoring", "setup")
_MODULES = {}


def load_module(name):
    if name not in SCHEMES:
        raise ValueError(f"Unknown scheme file: {name}")
    if name not in _MODULES:
        path = ROOT / (name + ".py")
        spec = importlib.util.spec_from_file_location("steane_comparison_" + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if module.SCHEME != SCHEMES[name][0]:
            raise ValueError(f"Unexpected SCHEME in {path.name}")
        _MODULES[name] = module
    return _MODULES[name]


class _Section:
    def __init__(self, meter, category):
        self.meter, self.category = meter, category

    def __enter__(self):
        m = self.meter
        tick = m.clock()
        if m.stack:
            m.nanoseconds[m.stack[-1]] += tick - m.last
        m.stack.append(self.category)
        m.calls[self.category] += 1
        # Do not charge stack/Counter bookkeeping to the timed child.  A new
        # clock read is necessary; reusing tick includes this Python overhead.
        m.last = m.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        m = self.meter
        tick = m.clock()
        assert m.stack[-1] == self.category
        m.nanoseconds[m.stack.pop()] += tick - m.last
        # Resume a parent only after meter bookkeeping finishes. In particular,
        # exiting a Stim/frame section must not inflate the control category.
        m.last = m.clock()
        return False

class TimingMeter:
    """Nested sections never charge a child's work to its parent category.

    Ledger bookkeeping between stop/resume clock reads is excluded. This still
    measures the instrumented Python implementation: context dispatch and clock
    read overhead are not magically removed; calibration is saved separately.
    """
    def __init__(self, clock=perf_counter_ns):
        self.clock = clock
        self.nanoseconds = Counter()
        self.calls = Counter()
        self.stack = []
        self.last = 0

    def section(self, category):
        if category not in CATEGORIES:
            raise ValueError(category)
        return _Section(self, category)

    def snapshot(self):
        assert not self.stack, "Cannot finalize active timing sections"
        return {"nanoseconds": {c: int(self.nanoseconds[c]) for c in CATEGORIES},
                "calls": {c: int(self.calls[c]) for c in CATEGORIES}}

def verify_meter():
    # Deterministic nesting oracle: parent 10+20+20, decoder 30, Stim 20.
    # The deliberately nonzero gaps between pause and resume represent timing
    # ledger work. None of them may be charged to a child or its parent.
    ticks = iter((0, 5, 15, 20, 50, 55, 75, 80, 100, 105, 125, 130))
    m = TimingMeter(clock=lambda: next(ticks))
    with m.section("control"):
        with m.section("decoder"):
            pass
        with m.section("stim"):
            pass
    assert m.nanoseconds == {"control": 50, "decoder": 30, "stim": 20}
    assert sum(m.nanoseconds.values()) == 100
    # Identical-category nesting must also be exclusive, not double counted.
    ticks = iter((0, 5, 15, 20, 40, 45, 65, 70))
    m = TimingMeter(clock=lambda: next(ticks))
    with m.section("control"):
        with m.section("control"):
            pass
    assert m.nanoseconds["control"] == 50
    # Exercise every category once so simulator/frame/setup/scoring boundaries
    # cannot silently fall back into the controller or decoder aggregate.
    ticks = iter(range(4 * len(CATEGORIES)))
    m = TimingMeter(clock=lambda: next(ticks))
    for category in CATEGORIES:
        with m.section(category):
            pass
    assert m.nanoseconds == {c: 1 for c in CATEGORIES}
    assert m.calls == {c: 1 for c in CATEGORIES}
    try:
        m.section("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown timing categories must be rejected")
    return {"status": "PASS", "exclusive_nesting": True,
            "same_category_nesting": True, "all_categories": True,
            "ledger_bookkeeping_excluded": True, "uses_stim": False}

def calibrate(repetitions=7, iterations=4096):
    empty, nested, payload = [], [], []
    for _ in range(repetitions):
        m = TimingMeter()
        tick = perf_counter_ns()
        for _ in range(iterations):
            with m.section("decoder"):
                pass
        empty.append((perf_counter_ns() - tick) / iterations)
        payload.append(m.nanoseconds["decoder"] / iterations)
        m = TimingMeter()
        tick = perf_counter_ns()
        for _ in range(iterations):
            with m.section("control"):
                with m.section("decoder"):
                    pass
        nested.append((perf_counter_ns() - tick) / iterations)
    return {"empty_section_wall_ns_median": median(empty),
            "nested_pair_wall_ns_median": median(nested),
            "empty_decoder_recorded_ns_median": median(payload),
            "iterations_per_repeat": iterations, "repeats": repetitions,
            "correction_applied_to_results": False}

def _timed(meter, category, function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with meter.section(category):
            return function(*args, **kwargs)
    return wrapped

def _section(meter, category):
    return meter.section(category) if meter is not None else nullcontext()

def warm_flag(scheme, meter=None):
    """Build/load deterministic recovery tables and all used gadget templates.

    No Stim simulation is run here. All work belongs to setup and is kept out
    of the warmed online decoder/control measurements.
    """
    with _section(meter, 'setup'):
        key = _scheme_key(scheme)
        module = _module(key)
        module.gadget('prep', 'zero')
        for generator in range(1, 7):
            module.gadget('bare', generator, f'scoring.g{generator}')
        if key == 'seq_flag':
            tables = module.sequential_maps()
            assert module.sequential_maps() is tables, 'Sequential LUT cache is not warm'
            for position, generator in enumerate(module.SCAN_ORDER, 1):
                module.gadget('flagged', generator, f'seq.resume.scan.g{generator}')
                for extra in module.SCAN_ORDER:
                    module.gadget('bare', extra, f'seq.resume.k{position}.followup.g{extra}')
        else:
            tables = module.local_maps(module.PROTOCOL_ID)
            assert module.local_maps(module.PROTOCOL_ID) is tables, 'Parallel LUT cache is not warm'
            for block in ('A', 'B'):
                module.gadget('block', block)
                for generator in range(1, 7):
                    module.gadget('bare', generator, f'sel.{block}.followup.g{generator}')
        cache = module.gadget.cache_info()
        assert cache.currsize < cache.maxsize, 'Gadget warmup exceeded its cache capacity'
        return {'scheme': scheme, 'status': 'PASS',
                'decoder_table_cache_identity': True,
                'gadget_cache_entries': cache.currsize,
                'gadget_cache_capacity': cache.maxsize,
                'uses_simulation': False}

def _engine_type(module, meter):
    class TimedEngine(module.StimEngine):
        def execute(self, kind, idx, *args):
            # Include gadget bookkeeping in the excluded physical backend
            # boundary, as its per-gate resource/fault bookkeeping is simulator
            # implementation work rather than the online QEC controller.
            with _section(meter, 'stim'):
                return super().execute(kind, idx, *args)

        def frame(self, idx, corrections):
            with _section(meter, 'scoring' if getattr(self, '_terminal', False) else 'frame'):
                return super().frame(idx, corrections)

        def terminal_readout(self, idx):
            with _section(meter, 'scoring'):
                self._terminal = True
                try:
                    return super().terminal_readout(idx)
                finally:
                    self._terminal = False

    return TimedEngine

def _prepare(module, p, n, seed, meter):
    with _section(meter, 'setup'):
        engine = _engine_type(module, meter)(n, p, seed)
    all_idx = np.arange(n, dtype=np.int64)
    ops, _ = module.gadget('prep', 'zero')
    accepted = np.ones(n, np.bool_)
    idx = all_idx
    # Input preparation and its acceptance precede entry into the QEC gadget.
    # They are outside the requested per-entered-QEC online timing boundary.
    with _section(meter, 'stim'):
        for name, targets, location_id in ops:
            value = engine.op(name, targets, location_id, idx)
            if location_id == 'prep.routeA.bareZ.M.ancilla':
                accepted[idx] = value == 0
                idx = all_idx[accepted]
    with _section(meter, 'scoring'):
        prep_x, prep_z = engine.frame_masks()
        prep_x &= 127
        prep_z &= 127
    return engine, all_idx, accepted, idx, prep_x, prep_z

def _finish(engine, all_idx, accepted, prep_x, prep_z, frame_x, frame_z,
            fallback_A, fallback_B, history_code, meter, **extra):
    with _section(meter, 'scoring'):
        out_x, out_z = engine.frame_masks()
        failed = np.zeros(engine.n, np.bool_)
        readout, _ = engine.terminal_readout(all_idx[accepted])
        failed[accepted] = readout
        logical_readout = np.full(engine.n, -1, np.int8)
        logical_readout[accepted] = readout
    return {
        'accepted': accepted, 'target_fail': failed, 'logical_readout': logical_readout,
        'Eout_x': out_x & 127, 'Eout_z': out_z & 127,
        'preparation_x': prep_x, 'preparation_z': prep_z,
        'frame_x': frame_x, 'frame_z': frame_z, 'resources': engine.resources,
        'fallback_A': fallback_A, 'fallback_B': fallback_B, 'history_code': history_code,
        'event_count': engine.event_count, 'prefix_survival_counts': engine.prefix,
        **extra,
    }

def _run_sequential(module, p, n, seed, meter):
    engine, all_idx, accepted, idx, prep_x, prep_z = _prepare(module, p, n, seed, meter)
    with _section(meter, 'setup'):
        corrections, fb = module.sequential_maps()
    with _section(meter, 'control'):
        frame_x = np.zeros(n, np.uint16)
        frame_z = np.zeros(n, np.uint16)
        fallback_A = np.zeros(n, np.uint8)
        fallback_B = np.zeros(n, np.uint8)
        history_code = np.full(n, -1, np.int32)
        trigger_count = np.zeros(n, np.uint8)
        trigger_mask = np.zeros(n, np.uint8)
        scan_pairs = np.full((n, 6), -1, np.int8)
        fresh_codes = np.full((n, 6), -1, np.int16)
        recovery_x = np.zeros((n, 6), np.uint16)
        recovery_z = np.zeros((n, 6), np.uint16)
        for position, g in enumerate(module.SCAN_ORDER, 1):
            if not len(idx):
                break
            syndrome, flag = engine.execute('flagged', idx, g, f'seq.resume.scan.g{g}')
            scan_pairs[idx, position - 1] = 2 * syndrome + flag
            trigger = syndrome | flag != 0
            selected = idx[trigger]
            if len(selected):
                # full_bare_vector packs the raw six reports but performs no
                # recovery lookup: its non-physical work stays in control.
                fresh = module.full_bare_vector(engine, selected, f'seq.resume.k{position}.followup')
                with _section(meter, 'decoder'):
                    branch = np.where(flag[trigger], g, 0)
                    c = corrections[branch, fresh]
                engine.frame(selected, c)
                frame_x[selected] ^= c[:, 0]
                frame_z[selected] ^= c[:, 1]
                fallback_B[selected] |= fb[branch, fresh]
                trigger_count[selected] += 1
                trigger_mask[selected] |= 1 << position - 1
                fresh_codes[selected, position - 1] = fresh
                recovery_x[selected, position - 1] = c[:, 0]
                recovery_z[selected, position - 1] = c[:, 1]
        history_code[idx] = trigger_mask[idx]
    # Resource/output accounting does not drive control or recovery and is
    # excluded from online categories.
    with _section(meter, 'scoring'):
        engine.resources[:, 8] = np.where(accepted, 36 + 24 * trigger_count.astype(np.uint16), 0)
        engine.resources[:, 9] = 6 + engine.resources[:, 8]
    return _finish(engine, all_idx, accepted, prep_x, prep_z, frame_x, frame_z,
                   fallback_A, fallback_B, history_code, meter,
                   trigger_count=trigger_count, trigger_mask=trigger_mask,
                   scan_pairs=scan_pairs, fresh_syndrome_codes=fresh_codes,
                   recovery_x=recovery_x, recovery_z=recovery_z)

def _followup(module, engine, idx, block, f, meter):
    # Follow-up choices depend on raw reports and belong to control.
    # Packing the correction-table index f*64+u is classified as decoder.
    with _section(meter, 'decoder'):
        local = f.astype(np.uint16) * 64
    for pattern, selected in module.SELECTED_GENERATORS[block].items():
        active = np.flatnonzero(f == int(pattern, 2))
        subset = idx[active]
        if not len(subset):
            continue
        u = np.zeros(len(subset), np.uint16)
        for g in selected:
            u = u * 2 + engine.execute('bare', subset, g, f'sel.{block}.followup.g{g}')[0]
        with _section(meter, 'decoder'):
            local[active] += u
    return local

def _run_parallel(module, p, n, seed, meter):
    engine, all_idx, accepted, idx, prep_x, prep_z = _prepare(module, p, n, seed, meter)
    with _section(meter, 'setup'):
        maps, fb, bx, bz, known = module.local_maps(module.PROTOCOL_ID)
    with _section(meter, 'control'):
        frame_x = np.zeros(n, np.uint16)
        frame_z = np.zeros(n, np.uint16)
        fallback_A = np.zeros(n, np.uint8)
        fallback_B = np.zeros(n, np.uint8)
        history_code = np.full(n, -1, np.int32)
        if len(idx):
            fA = module.binary(engine.execute('block', idx, 'A'))
            a = _followup(module, engine, idx, 'A', fA, meter)
            with _section(meter, 'decoder'):
                c = maps['A'][a]
            engine.frame(idx, c)
            frame_x[idx] ^= c[:, 0]
            frame_z[idx] ^= c[:, 1]
            fallback_A[idx] = fb['A'][a]
            fB = module.binary(engine.execute('block', idx, 'B'))
            b = _followup(module, engine, idx, 'B', fB, meter)
            with _section(meter, 'decoder'):
                h = a.astype(np.int32) * 512 + b
                c = np.column_stack((bx[h], bz[h]))
            engine.frame(idx, c)
            frame_x[idx] ^= c[:, 0]
            frame_z[idx] ^= c[:, 1]
            fallback_B[idx] = ~known[h]
            history_code[idx] = h
    return _finish(engine, all_idx, accepted, prep_x, prep_z, frame_x, frame_z,
                   fallback_A, fallback_B, history_code, meter)


def _scheme_key(scheme):
    return "seq_flag" if scheme in ("sequential_flag", "seq_flag") else "parallel"


def _module(scheme):
    return load_module("sequential_flag" if _scheme_key(scheme) == "seq_flag" else "parallel")


def execute(name, p, n, seed, meter=None):
    """Use identical physics and normalized results with or without timing."""
    module = load_module(name)
    if meter is None:
        return module.run_batch(p, n, seed)
    if name in ("parallel", "sequential_flag"):
        native = (_run_parallel if name == "parallel" else _run_sequential)(module, p, n, seed, meter)
        return module._normalize_result(native)
    # Only the online syndrome-to-mask function is wrapped. Preparation,
    # parity, controller decisions and terminal scoring never call this hook.
    original = module.correction_masks
    module.correction_masks = _timed(meter, "decoder", original)
    try:
        return module.run_batch(p, n, seed)
    finally:
        module.correction_masks = original


def prewarm(name, p_points):
    module = load_module(name)
    if name in ("parallel", "sequential_flag"):
        warm_flag(name)
        return
    fragments = [getattr(module, key) for key in ("zero_candidate", "cat_checked", "steane_zero_checked")
                 if hasattr(module, key)]
    if hasattr(module, "shor_readout"):
        fragments += [module.shor_readout(axis, support) for axis, support in module.GENERATORS]
    if hasattr(module, "steane_readout"):
        fragments += [module.steane_readout(axis) for axis in ("X", "Z")]
    for p in p_points:
        for fragment in fragments:
            module._compile(str(fragment), p)


def check_equivalence(names, seed):
    rows = []
    for name in names:
        for pi, p in enumerate((0.0, .001, .01)):
            run_seed = seed + 1000 * list(SCHEMES).index(name) + pi
            baseline = execute(name, p, 128, run_seed)
            timed = execute(name, p, 128, run_seed, TimingMeter())
            if baseline.keys() != timed.keys():
                raise AssertionError((name, p, "output keys differ"))
            for key in baseline:
                a, b = baseline[key], timed[key]
                if isinstance(a, np.ndarray):
                    equal = isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
                else:
                    equal = a == b
                if not equal:
                    raise AssertionError((name, p, key, "timing changed physical output"))
            rows.append({"scheme": name, "p": p, "seed": run_seed, "attempts_per_path": 128,
                         "fields_checked": len(baseline), "status": "PASS"})
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(paths):
    rows = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def plot_comparison(rows, path, metric):
    """Scientific SVG: log-p; log-probability or linear decoder time.

    Zero observed failure rates are upper-bound markers, never fabricated
    positive point estimates. Rows at p=0 are excluded from the logarithmic axis.
    """
    decoder = metric == "decoder"
    parsed, seen = [], set()
    for row in rows:
        p = float(row["p"])
        if p <= 0:
            continue
        key = "decoder_us_per_qec_mean" if decoder else "failure_rate"
        if row.get(key) in (None, ""):
            continue
        y = float(row[key])
        if decoder:
            delta = float(row["decoder_us_per_qec_sd"])
            low, high = max(0.0, y - delta), y + delta
        else:
            low, high = float(row["lower"]), float(row["upper"])
        if not all(math.isfinite(value) for value in (p, y, low, high)) or y < 0 or high < y:
            raise ValueError("Invalid plotted estimate or interval")
        if not decoder and y == 0 and high <= 0:
            raise ValueError("Zero estimates need a positive upper confidence bound")
        identity = (row["scheme"], p)
        if identity in seen:
            raise ValueError(f"Duplicate scheme/p point in input CSV files: {identity}")
        seen.add(identity)
        parsed.append({"scheme": row["scheme"], "label": row.get("label", row["scheme"]),
                       "p": p, "value": y, "low": low, "high": high,
                       "curve_metric": row.get("curve_metric", "")})
    if not parsed:
        raise ValueError("No finite positive-p points for this plot")
    order = [canonical for canonical, _ in SCHEMES.values()]
    names = sorted(set(row["scheme"] for row in parsed), key=lambda name: order.index(name) if name in order else len(order))
    xmin, xmax = math.log10(min(row["p"] for row in parsed)), math.log10(max(row["p"] for row in parsed))
    if xmin == xmax:
        xmin, xmax = xmin - .5, xmax + .5
    if decoder:
        ymin, ymax = 0.0, max(max(row["high"] for row in parsed) * 1.12, 1e-6)
    else:
        smallest = min(row[key] for row in parsed for key in ("value", "low", "high") if row[key] > 0)
        ymin = math.floor(math.log10(smallest / 1.5))
        ymax = math.ceil(math.log10(max(row["high"] for row in parsed) * 1.2))
        if ymin == ymax:
            ymax += 1
    left, right, top, bottom = 88, 805, 75, 505
    tx = lambda p: left + (math.log10(p) - xmin) / (xmax - xmin) * (right - left)
    ty = lambda y: bottom - ((y if decoder else math.log10(max(y, 10 ** ymin))) - ymin) / (ymax - ymin) * (bottom - top)
    title = "Pure decoder time per entered QEC" if decoder else "Steane QEC logical error comparison"
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="615" viewBox="0 0 1060 615">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" font-size="13" fill="#17263b">',
             f'<text x="88" y="31" font-size="23">{title}</text>',
             f'<text x="88" y="53" fill="#536277">{"Mean +/- sample SD across repetitions; batch-amortized Python time" if decoder else "CSV estimates and supplied confidence intervals; completed-output denominator"}</text>']
    ticks_y = [ymax * i / 5 for i in range(6)] if decoder else [10 ** power for power in range(int(ymin), int(ymax) + 1)]
    for value in ticks_y:
        parts += [f'<path d="M {left} {ty(value):.3f} H {right}" stroke="#e3e8f0"/>',
                  f'<text x="{left-12}" y="{ty(value)+5:.3f}" text-anchor="end">{value:.3g}</text>']
    ticks_x = [10 ** power for power in range(math.ceil(xmin), math.floor(xmax) + 1)]
    if len(ticks_x) < 2:
        ticks_x = sorted(set(row["p"] for row in parsed))
    for value in ticks_x:
        parts += [f'<path d="M {tx(value):.3f} {top} V {bottom}" stroke="#e3e8f0"/>',
                  f'<text x="{tx(value):.3f}" y="{bottom+25}" text-anchor="middle">{value:g}</text>']
    colors = ("#0288d1", "#1565c0", "#7b1fa2", "#ef6c00", "#c79100", "#008577", "#bd5696")
    any_zero = False
    for i, name in enumerate(names):
        group = sorted((row for row in parsed if row["scheme"] == name), key=lambda row: row["p"])
        color = colors[(order.index(name) if name in order else i) % len(colors)]
        # Break connecting lines at zero-rate upper-bound observations.
        segment = []
        for row in [*group, None]:
            if row is None or (not decoder and row["value"] == 0):
                if len(segment) > 1:
                    points = " ".join(f'{tx(item["p"]):.3f},{ty(item["value"]):.3f}' for item in segment)
                    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
                segment = []
            else:
                segment.append(row)
        for row in group:
            x = tx(row["p"])
            if not decoder and row["value"] == 0:
                any_zero = True
                y = ty(row["high"])
                parts.append(f'<path d="M {x-4:.3f} {y-3:.3f} L {x+4:.3f} {y-3:.3f} L {x:.3f} {y+5:.3f} Z" fill="{color}"/>')
            else:
                y = ty(row["value"])
                lo, hi = ty(row["low"]), ty(row["high"])
                parts += [f'<path d="M {x:.3f} {lo:.3f} V {hi:.3f} M {x-3:.3f} {hi:.3f} h 6" stroke="{color}"/>',
                          f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3.5" fill="{color}"/>']
        parts += [f'<path d="M 830 {100+i*31} h 25" stroke="{color}" stroke-width="3"/>',
                  f'<text x="865" y="{105+i*31}">{escape(group[0]["label"])}</text>']
    parts += [f'<path d="M {left} {top} V {bottom} H {right}" fill="none" stroke="#17263b"/>',
              '<text x="365" y="555">Physical noise parameter p (log scale)</text>',
              f'<text transform="translate(23,365) rotate(-90)">{"microseconds / entered QEC" if decoder else "Conditional error probability (log scale)"}</text>']
    metrics = {row["curve_metric"] for row in parsed}
    metric_notes = {"L": "All plotted curves use L (logical-zero target failure after ideal SteaneCode scoring).",
                    "B": "All plotted curves use B (actual output outside recovery radius one).",
                    "L_or_B": "All plotted curves use L OR B (recovery-inclusive score)."}
    footer = ("Excludes control, Stim, frame, scoring and setup; entered inputs include later QEC aborts." if decoder else
              metric_notes[next(iter(metrics))] if len(metrics) == 1 and next(iter(metrics)) in metric_notes else
              "Read curve_metric in CSV: retained Steane-1 uses L OR B; the other retained curves use L.")
    parts.append(f'<text x="88" y="582" font-size="12" fill="#536277">{footer}</text>')
    if any_zero:
        parts.append('<text x="88" y="601" font-size="12" fill="#536277">Downward triangles show supplied upper bounds when zero failures were observed; p=0 is omitted.</text>')
    elif not decoder:
        parts.append('<text x="88" y="601" font-size="12" fill="#536277">p=0 is omitted; confidence intervals with a zero lower bound are clipped at the plot floor.</text>')
    parts.append('</g></svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def summarize(repetitions):
    result = []
    for scheme, p in dict.fromkeys((row["scheme"], row["p"]) for row in repetitions):
        rows = [row for row in repetitions if row["scheme"] == scheme and row["p"] == p]
        values = [row["decoder_us_per_qec"] for row in rows]
        result.append({"scheme": scheme, "label": rows[0]["label"], "p": p,
                       "repeats": len(rows), "qec_started": sum(row["qec_started"] for row in rows),
                       "qec_completed": sum(row["qec_completed"] for row in rows),
                       "decoder_us_per_qec_mean": mean(values), "decoder_us_per_qec_sd": stdev(values),
                       "decoder_total_ns": sum(row["decoder_ns"] for row in rows),
                       "decoder_total_calls": sum(row["decoder_calls"] for row in rows)})
    return result


def benchmark(args, output):
    verify_meter()
    for name in args.schemes:
        prewarm(name, args.p)
    print("Checking same-seed timed/untimed physical equality...", flush=True)
    checks = check_equivalence(args.schemes, args.seed)
    write_csv(output / "verification.csv", checks)
    calibration = calibrate()
    for name in args.schemes:
        si = list(SCHEMES).index(name)
        for pi, p in enumerate(args.p):
            execute(name, p, args.warmup, args.seed + 100000 + si * 100 + pi, TimingMeter())
    repetitions = []
    for repeat in range(args.repetitions):
        jobs = [(name, pi, p) for name in args.schemes for pi, p in enumerate(args.p)]
        random.Random(args.seed + repeat).shuffle(jobs)
        for name, pi, p in jobs:
            si = list(SCHEMES).index(name)
            seed = args.seed + 1000000 + repeat * 10000 + si * 100 + pi
            meter = TimingMeter()
            started = perf_counter_ns()
            data = execute(name, p, args.batch, seed, meter)
            elapsed = perf_counter_ns() - started
            timing = meter.snapshot()
            accepted = int(np.count_nonzero(data["accepted"]))
            completed = int(np.count_nonzero(data["completed"]))
            if accepted == 0:
                raise RuntimeError(f"{name}: zero accepted inputs; increase --batch")
            if sum(timing["nanoseconds"].values()) > elapsed:
                raise AssertionError("Timing categories overlap")
            repetitions.append({"scheme": SCHEMES[name][0], "label": SCHEMES[name][1], "p": p,
                                "repeat": repeat, "seed": seed, "attempted": args.batch,
                                "qec_started": accepted, "qec_completed": completed,
                                "decoder_ns": timing["nanoseconds"]["decoder"],
                                "decoder_calls": timing["calls"]["decoder"],
                                "decoder_us_per_qec": timing["nanoseconds"]["decoder"] / accepted / 1000})
        write_csv(output / "decoder_repetitions.csv", repetitions)
        print(f"Finished decoder repetition {repeat+1}/{args.repetitions}", flush=True)
    summary = summarize(repetitions)
    write_csv(output / "decoder_time.csv", summary)
    plot_comparison(summary, output / "decoder_time.svg", "decoder")
    return {"timing_rows": len(repetitions), "attempts": sum(row["attempted"] for row in repetitions),
            "equivalence_cases": len(checks), "calibration": calibration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "plot", "decoder"), default="all")
    parser.add_argument("--schemes", nargs="+", choices=tuple(SCHEMES), default=list(SCHEMES))
    parser.add_argument("--p", nargs="+", type=float, default=[1e-4, .001, .01])
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026102201)
    parser.add_argument("--historical-plan", action="store_true", help="Use the original five p values, batch 8192, nine repetitions, warmup 1024")
    parser.add_argument("--logical-csv", nargs="+", type=Path, help="Final CSV or one/more fresh per-scheme CSV files")
    parser.add_argument("--decoder-csv", type=Path, help="Saved decoder CSV to redraw in plot mode")
    parser.add_argument("--output", type=Path, help="New output directory; never overwrite")
    args = parser.parse_args()
    if args.historical_plan:
        args.p, args.batch, args.repetitions, args.warmup = [1e-4, 3e-4, .001, .003, .01], 8192, 9, 1024
    if args.batch < 1 or args.warmup < 1 or args.repetitions < 2 or args.seed < 0:
        parser.error("batch/warmup must be positive, repetitions >=2 and seed >=0")
    if len(set(args.p)) != len(args.p) or len(set(args.schemes)) != len(args.schemes):
        parser.error("Repeated p values or schemes are not allowed")
    if any(not math.isfinite(p) or not 0 <= p <= .1 for p in args.p):
        parser.error("p must be finite and between 0 and 0.1")
    if args.mode != "plot" and not any(p > 0 for p in args.p):
        parser.error("Include at least one p>0 for the logarithmic comparison plot")
    output = args.output or PROJECT_ROOT / "output" / ("comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *[ROOT / (name + ".py") for name in args.schemes]]
    lines = ["Steane QEC comparison", f"Python: {sys.version}", f"Platform: {platform.platform()}",
             f"Stim: {stim.__version__}; NumPy: {np.__version__}", f"Mode: {args.mode}",
             f"Schemes: {', '.join(args.schemes)}", f"p: {args.p}", f"Batch: {args.batch}",
             f"Repetitions: {args.repetitions}; warmup: {args.warmup}; seed: {args.seed}",
             "Metric: pure decoder nanoseconds / entered QEC inputs (including later aborts).",
             "Excludes raw parity/packing, adaptive control, Stim, frame, scoring and setup.",
             "Error bars: sample SD across repetitions, not shot confidence intervals or hardware latency.",
             "Timing overhead calibration is recorded without subtraction.", "Source SHA256:"]
    lines += [f"  {path.name}: {hashlib.sha256(path.read_bytes()).hexdigest()}" for path in sources]
    (output / "run_info.txt").write_text("\n".join(lines) + "\nStatus: RUNNING\n", encoding="utf-8")
    try:
        if args.mode in ("all", "plot"):
            logical_paths = args.logical_csv or [PROJECT_ROOT / "results/logical_error.csv"]
            plot_comparison(read_csv(logical_paths), output / "logical_error.svg", "logical")
            if args.mode == "plot":
                decoder_path = args.decoder_csv or PROJECT_ROOT / "results/decoder_time.csv"
                plot_comparison(read_csv([decoder_path]), output / "decoder_time.svg", "decoder")
            lines += ["Logical CSV: " + ", ".join(str(path.resolve()) for path in logical_paths)]
        if args.mode in ("all", "decoder"):
            report = benchmark(args, output)
            lines += [f"{key}: {value}" for key, value in report.items()]
        lines.append("Status: COMPLETE")
    except Exception as error:
        lines.append(f"Status: FAILED ({error!r})")
        raise
    finally:
        (output / "run_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved comparison: {output.resolve()}")


if __name__ == "__main__":
    main()
