"""Standalone Seq-flag Steane [[7,1,3]] QEC experiment.

Contains the full circuit templates, simulator, decoder and physical protocol.
Runtime dependencies: Python standard library, NumPy and Stim only.
No other local source, decoder table, JSON or NPZ file is required.

The workflow is noisy logical-zero preparation, bare logical-Z verification,
adaptive QEC, then ideal terminal SteaneCode recovery for scoring. The terminal
scoring operations are not part of the noisy QEC or its resource count.
"""
from __future__ import annotations

SCHEME = 'sequential_flag'
LABEL = 'Seq-flag'
DEFAULT_METRIC = "target"
SOURCE_CORE_SHA256 = "1d962a6f26689bd37eeab709ea0bb1a33d55f38f0c3838dde917dabeb9fd4f29"

import hashlib
import math
import itertools
from functools import lru_cache

import numpy as np
import stim

CANONICAL_GENERATORS = ((85, 0), (0, 102), (0, 120), (0, 85), (102, 0), (120, 0))

DATA_QUBITS = tuple(range(7))

GENERATORS = {1: ('X', (0, 2, 4, 6)), 2: ('Z', (1, 2, 5, 6)), 3: ('Z', (3, 4, 5, 6)), 4: ('Z', (0, 2, 4, 6)), 5: ('X', (1, 2, 5, 6)), 6: ('X', (3, 4, 5, 6))}

GENERATOR_SPECS = GENERATORS

def syndrome(x, z):
    return tuple(((gx & z).bit_count() + (gz & x).bit_count() & 1 for gx, gz in CANONICAL_GENERATORS))

def ordinary_correction(sigma):
    s = tuple(map(int, sigma))
    qx = s[3] + 2 * s[1] + 4 * s[2]
    qz = s[0] + 2 * s[4] + 4 * s[5]
    return (0 if not qx else 1 << qx - 1, 0 if not qz else 1 << qz - 1)

@lru_cache(maxsize=1)
def sequential_maps():
    correction = np.zeros((7, 64, 2), np.uint16)
    fallback = np.ones((7, 64), np.uint8)
    ordinary_domain = {syndrome(0, 0)}
    for q in range(7):
        ordinary_domain.update((syndrome(x << q, z << q) for x, z in ((1, 0), (1, 1), (0, 1))))
    for g in range(7):
        hook, flag_domain = ({}, set())
        axis, support = GENERATORS.get(g, ('X', (0, 0, 0, 0)))
        if g:
            a, b, c, d = support
            axis_pair = lambda mask: (mask, 0) if axis == 'X' else (0, mask)
            for mask, recovery in ((0, 0), (1 << d, 1 << d), (1 << c | 1 << d, 1 << c | 1 << d), (1 << b | 1 << c | 1 << d, 1 << a)):
                s = syndrome(*axis_pair(mask))
                component = (s[3], s[1], s[2]) if axis == 'X' else (s[0], s[4], s[5])
                hook[component] = axis_pair(recovery)
            errors = [axis_pair(mask) for mask in (0, 1 << d, 1 << c | 1 << d, 1 << b | 1 << c | 1 << d)]
            for q, mask in ((b, 1 << c | 1 << d), (c, 1 << d)):
                x, z = axis_pair(mask)
                opposite = (0, 1 << q) if axis == 'X' else (1 << q, 0)
                same = axis_pair(1 << q)
                errors.extend(((x ^ opposite[0], z ^ opposite[1]), (x ^ opposite[0] ^ same[0], z ^ opposite[1] ^ same[1])))
            flag_domain = {syndrome(x, z) for x, z in errors}
        for code in range(64):
            s = tuple(map(int, format(code, '06b')))
            x, z = ordinary_correction(s)
            if g:
                component = (s[3], s[1], s[2]) if axis == 'X' else (s[0], s[4], s[5])
                hx, hz = hook.get(component, (x, 0) if axis == 'X' else (0, z))
                correction[g, code] = (hx, z) if axis == 'X' else (x, hz)
                fallback[g, code] = s not in flag_domain
            else:
                correction[g, code] = (x, z)
                fallback[g, code] = s not in ordinary_domain
    return (correction, fallback)

PROTOCOL_ID = 'sequential_resume'

PROTOCOLS = (PROTOCOL_ID,)

DISPLAY_NAME = 'sequential flag'

DATA_NAME = 'sequential_flag'

DEFAULT_STATE, DEFAULT_P, DEFAULT_ATTEMPTS = 'zero', 0.001, 10000

MASTER_SEED = DEFAULT_SEED = 2026101323

NOISE_MODEL = {'post_reset': 10, 'post_1q': 1, 'post_2q': 5, 'report_flip': 2, 'idle': 0}

ROUTE_A_CX = ((1, 0), (3, 5), (2, 6), (1, 4), (2, 0), (3, 6), (1, 5), (6, 4))

LOGICAL_Z_SUPPORT = (0, 5, 6)

PAULI_BITS = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}

PAULI1 = ('X', 'Y', 'Z')

PAULI2 = tuple((a + b for a, b in itertools.product('IXYZ', repeat=2) if a + b != 'II'))

RATES, SYNDROME_ANCILLA, FLAG_ANCILLA = (1, 2, 5, 10), 7, 8

RESOURCE_NAMES = ('CX_total', 'reset_total', 'measurement_total', 'qec_generators', 'QEC_CX', 'QEC_reset', 'QEC_measurement', 'primitive_total', 'QEC_CNOT_depth', 'CNOT_depth_total')

def location_stage(location_id):
    return 'qec' if not location_id.startswith('prep.') else 'verification' if 'bareZ' in location_id else 'suffix' if '.suffix.' in location_id else 'encoding'

SCAN_ORDER = (6, 5, 1, 3, 2, 4)

FRESH_CANONICAL_INDICES = (2, 4, 3, 5, 1, 0)

@lru_cache(maxsize=256)
def gadget(kind, *args):
    ops, generators = [], 1
    def add(name, targets, location):
        ops.append((name, tuple(targets), location))
    if kind == 'prep':
        add('R', (7,), 'prep.routeA.bareZ.R.ancilla')
        for q in DATA_QUBITS: add('R', (q,), f'prep.routeA.R.q{q}')
        for q in (1, 2, 3): add('H', (q,), f'prep.routeA.H.q{q}')
        for i, targets in enumerate(ROUTE_A_CX, 1): add('CX', targets, f'prep.routeA.CX.{i:02d}')
        for i, q in enumerate(LOGICAL_Z_SUPPORT, 1): add('CX', (q, 7), f'prep.routeA.bareZ.CX.{i:02d}')
        add('M', (7,), 'prep.routeA.bareZ.M.ancilla')
    elif kind in ('bare', 'flagged'):
        g, prefix = args
        axis, support = GENERATOR_SPECS[g]
        prepare, measure = ('RX', 'MX') if axis == 'X' else ('R', 'M')
        add(prepare, (7,), f'{prefix}.{prepare}.syndrome')
        if kind == 'flagged':
            flag_prepare, flag_measure = ('R', 'M') if axis == 'X' else ('RX', 'MX')
            add(flag_prepare, (8,), f'{prefix}.{flag_prepare}.flag')
            a, b, c, d = support
            gates = ((7, a), (7, 8), (7, b), (7, c), (7, 8), (7, d)) if axis == 'X' else ((a, 7), (8, 7), (b, 7), (c, 7), (8, 7), (d, 7))
        else:
            gates = tuple((7, q) if axis == 'X' else (q, 7) for q in support)
        for i, targets in enumerate(gates, 1): add('CX', targets, f'{prefix}.CX.{i:02d}')
        add(measure, (7,), f'{prefix}.{measure}.syndrome')
        if kind == 'flagged': add(flag_measure, (8,), f'{prefix}.{flag_measure}.flag')
    else: raise ValueError(kind)
    return tuple(ops), generators

@lru_cache(maxsize=1024)
def descriptor(name, targets, location_id):
    kind = 'post_reset' if name in ('R', 'RX') else 'post_2q' if name == 'CX' else 'report_flip' if name in ('M', 'MX') else 'post_1q'
    support = PAULI2 if name == 'CX' else ('flip',) if name in ('M', 'MX') else PAULI1
    masks = [(sum((PAULI_BITS[p][0] << q for q, p in zip(targets, pauli))), sum((PAULI_BITS[p][1] << q for q, p in zip(targets, pauli)))) if pauli != 'flip' else (0, 0) for pauli in support]
    dx, dz = zip(*masks)
    return (NOISE_MODEL[kind], support, np.array(dx, np.uint16), np.array(dz, np.uint16))

@lru_cache(maxsize=64)
def instruction(name, targets, probability=None):
    return stim.CircuitInstruction(name, list(targets), [] if probability is None else [probability])

class StimEngine:

    def __init__(self, n, p, seed, forced=None, record_events=False):
        self.n, self.p, self.seed, self.seed_counter = (n, float(p), seed, 0)
        self.sim = self._new_sim(n)
        self.active_sim, self.active_idx = (self.sim, np.arange(n))
        self.resources = np.zeros((n, len(RESOURCE_NAMES)), np.uint16)
        self.event_count, self.applied = (np.zeros(n, np.uint16), np.zeros(n, np.uint8))
        self.prefix = np.zeros((n, 4), np.uint16)
        self.forced, self.record_events = (forced, record_events)
        self.events = [[] for _ in range(n)] if record_events else None
        self.measurements = [[] for _ in range(n)] if record_events else None

    def _new_sim(self, n):
        digest = hashlib.sha256(f'stim-native|{self.seed}|{self.seed_counter}'.encode('ascii')).digest()
        self.seed_counter += 1
        return stim.FlipSimulator(batch_size=n, num_qubits=10, disable_stabilizer_randomization=True, seed=int.from_bytes(digest[:8], 'big'))

    def _commit(self):
        if self.active_sim is self.sim: return
        old_x, old_z, *_ = self.sim.to_numpy(output_xs=True, output_zs=True)
        new_x, new_z, *_ = self.active_sim.to_numpy(output_xs=True, output_zs=True)
        for pauli, old, new in [('X', old_x, new_x), ('Z', old_z, new_z)]:
            mask = np.zeros((10, self.n), np.bool_)
            mask[:, self.active_idx] = old[:, self.active_idx] ^ new
            self.sim.broadcast_pauli_errors(pauli=pauli, mask=mask)
        self.active_sim, self.active_idx = (self.sim, np.arange(self.n))

    def _activate(self, idx):
        if np.array_equal(idx, self.active_idx): return self.active_sim
        self._commit()
        if len(idx) == self.n: return self.sim
        xs, zs, *_ = self.sim.to_numpy(output_xs=True, output_zs=True)
        sim = self._new_sim(len(idx))
        sim.broadcast_pauli_errors(pauli='X', mask=np.ascontiguousarray(xs[:, idx]))
        sim.broadcast_pauli_errors(pauli='Z', mask=np.ascontiguousarray(zs[:, idx]))
        self.active_sim, self.active_idx = (sim, idx.copy())
        return sim

    def frame_masks(self):
        self._commit()
        xs, zs, *_ = self.sim.to_numpy(output_xs=True, output_zs=True)
        powers = (1 << np.arange(10, dtype=np.uint16))[:, None]
        return ((xs * powers).sum(axis=0).astype(np.uint16), (zs * powers).sum(axis=0).astype(np.uint16))

    def _record_fault(self, idx, targets, dx, dz, location_id):
        changed = np.any(dx | dz, axis=0)
        where = idx[changed]
        self.event_count[where] += 1
        if self.record_events:
            for local, row in zip(np.flatnonzero(changed), where):
                text = ''.join(('IXZY'[int(dx[q, local]) + 2 * int(dz[q, local])] for q in targets))
                self.events[int(row)].append((location_id, text))

    def _child(self, sim, rows, xs=None, zs=None):
        if xs is None: xs, zs, *_ = sim.to_numpy(output_xs=True, output_zs=True)
        child = self._new_sim(len(rows))
        child.broadcast_pauli_errors(pauli='X', mask=np.ascontiguousarray(xs[:, rows]))
        child.broadcast_pauli_errors(pauli='Z', mask=np.ascontiguousarray(zs[:, rows]))
        return (child, xs, zs)

    def _merge(self, sim, child, rows, old_x, old_z):
        new_x, new_z, *_ = child.to_numpy(output_xs=True, output_zs=True)
        dx, dz = (old_x[:, rows] ^ new_x, old_z[:, rows] ^ new_z)
        for pauli, difference in [('X', dx), ('Z', dz)]:
            mask = np.zeros((10, sim.batch_size), np.bool_)
            mask[:, rows] = difference
            sim.broadcast_pauli_errors(pauli=pauli, mask=mask)
        return (dx, dz)

    def _depolarize(self, sim, idx, rows, name, targets, probability, location_id):
        if not len(rows):
            sim.do(instruction(name, targets, 0.0))
            return
        full = len(rows) == len(idx)
        target_sim, before_x, before_z = (sim, None, None) if full else self._child(sim, rows)
        if probability:
            if full: before_x, before_z, *_ = sim.to_numpy(output_xs=True, output_zs=True)
        target_sim.do(instruction(name, targets, probability))
        if probability:
            if full:
                after_x, after_z, *_ = sim.to_numpy(output_xs=True, output_zs=True)
                dx, dz = (before_x ^ after_x, before_z ^ after_z)
            else:
                dx, dz = self._merge(sim, target_sim, rows, before_x, before_z)
            self._record_fault(idx[rows], targets, dx, dz, location_id)

    def op(self, name, targets, location_id, idx):
        if not len(idx): return np.empty(0, np.uint8) if name in ('M', 'MX') else None
        sim = self._activate(idx)
        qec = location_stage(location_id) == 'qec'
        self.resources[idx, 7] += 1
        column = 1 if name in ('R', 'RX') else 0 if name == 'CX' else 2 if name in ('M', 'MX') else None
        if column is not None:
            self.resources[idx, column] += 1
            self.resources[idx, column + 4] += qec
        factor, support, dx, dz = descriptor(name, targets, location_id)
        hit = np.full(len(idx), -1, np.int16)
        if self.forced is not None:
            before = self.applied[idx].copy()
            code = self.forced['id_codes'].get(location_id, -1) if 'id_codes' in self.forced else location_id
            hit1 = (self.forced['first_id'][idx] == code) & (before == 0)
            hit2 = (self.forced['second_id'][idx] == code) & (before == 1)
            hit[hit1] = self.forced['first_pauli'][idx[hit1]]
            hit[hit2] = self.forced['second_pauli'][idx[hit2]]
            self.applied[idx[hit1]] += 1
            self.applied[idx[hit2]] += 1
            survival = (before < 2) & ~hit1 & ~hit2
            self.prefix[idx[survival], RATES.index(factor)] += 1
            noisy = before == 2
        else:
            noisy = np.ones(len(idx), np.bool_)
        if name in ('M', 'MX'):
            xs, zs, *_ = sim.to_numpy(output_xs=True, output_zs=True)
            true = (xs if name == 'M' else zs)[targets[0]].astype(np.uint8)
            value = np.zeros(len(idx), np.uint8)
            groups = [(noisy, factor * self.p), (~noisy & (hit >= 0), 1.0), (~noisy & (hit < 0), 0.0)]
            for selected, probability in groups:
                rows = np.flatnonzero(selected)
                if len(rows):
                    full = len(rows) == len(idx)
                    target_sim = sim if full else self._child(sim, rows, xs, zs)[0]
                    target_sim.do(instruction(name, targets, probability))
                    value[rows] = target_sim.get_measurement_flips(record_index=-1)
                    if not full: self._merge(sim, target_sim, rows, xs, zs)
            changed = value != true
            self.event_count[idx[changed]] += 1
            if self.record_events:
                for row in idx[changed]:
                    self.events[int(row)].append((location_id, 'flip'))
                for row, t, v in zip(idx, true, value):
                    self.measurements[int(row)].append((location_id, int(t), int(v)))
            return value
        sim.do(instruction(name, targets))
        noise_gate = 'DEPOLARIZE2' if name == 'CX' else 'DEPOLARIZE1'
        self._depolarize(sim, idx, np.flatnonzero(noisy), noise_gate, targets, factor * self.p, location_id)
        chosen = np.flatnonzero(hit >= 0)
        if len(chosen):
            for pauli, masks in [('X', dx), ('Z', dz)]:
                mask = np.zeros((10, len(idx)), np.bool_)
                mask[:, chosen] = (masks[hit[chosen], None] >> np.arange(10) & 1).T.astype(np.bool_)
                sim.broadcast_pauli_errors(pauli=pauli, mask=mask)
            self.event_count[idx[chosen]] += 1
            if self.record_events:
                for row, which in zip(idx[chosen], hit[chosen]):
                    self.events[int(row)].append((location_id, support[int(which)]))
        return None

    def execute(self, kind, idx, *args):
        ops, generators = gadget(kind, *args)
        if kind != 'prep': self.resources[idx, 3] += generators
        reports = []
        for name, targets, location_id in ops:
            value = self.op(name, targets, location_id, idx)
            if value is not None: reports.append(value)
        return reports

    def frame(self, idx, corrections):
        if not len(idx): return
        sim = self._activate(idx)
        for column, pauli in enumerate(('X', 'Z')):
            mask = (corrections[:, column, None] >> np.arange(10) & 1).T.astype(np.bool_)
            sim.broadcast_pauli_errors(pauli=pauli, mask=mask)

    def terminal_readout(self, idx):
        if not len(idx): return (np.empty(0, np.bool_), np.empty((0, 6), np.uint8))
        sim = self._activate(idx)
        reports = []
        for g in range(1, 7):
            ops, _ = gadget('bare', g, f'scoring.g{g}')
            circuit = stim.Circuit()
            for name, targets, _ in ops:
                circuit.append(name, targets)
            sim.do(circuit)
            reports.append(sim.get_measurement_flips(record_index=-1).astype(np.uint16))
        s = np.array(reports).T
        qx, qz = (s[:, 3] + 2 * s[:, 1] + 4 * s[:, 2], s[:, 0] + 2 * s[:, 4] + 4 * s[:, 5])
        c = np.column_stack((np.where(qx, 1 << np.maximum(qx, 1) - 1, 0), np.where(qz, 1 << np.maximum(qz, 1) - 1, 0)))
        self.frame(idx, c)
        sim.do(stim.Circuit('MPP Z0*Z5*Z6'))
        return (sim.get_measurement_flips(record_index=-1).copy(), s.astype(np.uint8))

def binary(reports):
    result = np.zeros(len(reports[0]), np.uint16)
    for value in reports:
        result = result * 2 + value
    return result

def full_bare_vector(engine, idx, prefix):
    reports = [engine.execute('bare', idx, g, f'{prefix}.g{g}')[0] for g in SCAN_ORDER]
    return binary([reports[i] for i in FRESH_CANONICAL_INDICES])

def _native_run_batch(state, protocol, p, n, seed, forced=None, record_events=False):
    if state != 'zero' or protocol not in PROTOCOLS: raise ValueError((state, protocol))
    engine = StimEngine(n, p, seed, forced, record_events)
    all_idx = np.arange(n, dtype=np.int64)
    ops, _ = gadget('prep', state)
    accepted = np.ones(n, np.bool_)
    idx = all_idx
    for name, targets, location_id in ops:
        value = engine.op(name, targets, location_id, idx)
        if location_id == 'prep.routeA.bareZ.M.ancilla':
            accepted[idx] = value == 0
            idx = all_idx[accepted]
    prep_x, prep_z = engine.frame_masks()
    prep_x &= 127
    prep_z &= 127
    frame_x = np.zeros(n, np.uint16)
    frame_z = np.zeros(n, np.uint16)
    fallback_A = np.zeros(n, np.uint8)
    fallback_B = np.zeros(n, np.uint8)
    history_code = np.full(n, -1, np.int32)
    corrections, fb = sequential_maps()
    trigger_count = np.zeros(n, np.uint8)
    trigger_mask = np.zeros(n, np.uint8)
    scan_pairs = np.full((n, 6), -1, np.int8)
    fresh_codes = np.full((n, 6), -1, np.int16)
    recovery_x = np.zeros((n, 6), np.uint16)
    recovery_z = np.zeros((n, 6), np.uint16)
    for position, g in enumerate(SCAN_ORDER, 1):
        if not len(idx): break
        syndrome, flag = engine.execute('flagged', idx, g, f'seq.resume.scan.g{g}')
        scan_pairs[idx, position - 1] = 2 * syndrome + flag
        trigger = syndrome | flag != 0
        selected = idx[trigger]
        if len(selected):
            fresh = full_bare_vector(engine, selected, f'seq.resume.k{position}.followup')
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
    engine.resources[:, 8] = np.where(accepted, 36 + 24 * trigger_count.astype(np.uint16), 0)
    engine.resources[:, 9] = 6 + engine.resources[:, 8]
    out_x, out_z = engine.frame_masks()
    failed = np.zeros(n, np.bool_)
    readout, terminal_syndrome = engine.terminal_readout(all_idx[accepted])
    failed[accepted] = readout
    logical_readout = np.full(n, -1, np.int8)
    logical_readout[accepted] = readout
    if forced is not None and (not np.all(engine.applied == forced.get('expected_count', 2))): raise AssertionError('A forced proposal was not reached exactly')
    output = {'accepted': accepted, 'target_fail': failed, 'logical_readout': logical_readout, 'Eout_x': out_x & 127, 'Eout_z': out_z & 127, 'preparation_x': prep_x, 'preparation_z': prep_z, 'frame_x': frame_x, 'frame_z': frame_z, 'resources': engine.resources, 'fallback_A': fallback_A, 'fallback_B': fallback_B, 'history_code': history_code, 'event_count': engine.event_count, 'prefix_survival_counts': engine.prefix, 'trigger_count': trigger_count, 'trigger_mask': trigger_mask, 'scan_pairs': scan_pairs, 'fresh_syndrome_codes': fresh_codes, 'recovery_x': recovery_x, 'recovery_z': recovery_z}
    if record_events: output.update(events=engine.events, measurements=engine.measurements)
    return output


def terminal_oracle(x, z):
    """Independent Hamming-SteaneCode logical-zero target score, low-bit qubit order."""
    del z
    x = np.asarray(x, dtype=np.uint16)
    s = [np.bitwise_count(x & mask) & 1 for mask in (85, 102, 120)]
    address = s[0] + 2 * s[1] + 4 * s[2]
    correction = np.where(address, np.uint16(1) << (np.maximum(address, 1) - 1), 0)
    return (np.bitwise_count((x ^ correction) & 97) & 1).astype(bool)


def endpoint_outside_radius_one(x, z):
    """Distance to any valid Steane codeword exceeds one, before scoring.

    This is a syndrome-class radius test, not a trajectory-wise certification
    of fault tolerance based on the number of physical faults.
    """
    x = np.asarray(x, dtype=np.uint16)
    z = np.asarray(z, dtype=np.uint16)
    def labels(mask):
        bits = [np.bitwise_count(mask & support) & 1 for support in (85, 102, 120)]
        return bits[0] | (bits[1] << 1) | (bits[2] << 2)
    a, b = labels(x), labels(z)
    return (a != 0) & (b != 0) & (a != b)


def run_batch(p, n, seed):
    """Run n physical attempts with logical-zero input.

    Status 0 means preparation rejected; status 2 means QEC completed. The
    failure fields are false placeholders on rejected attempts. Eout_x/z
    describe actual QEC output before ideal terminal SteaneCode scoring. Recovery
    is a virtual Pauli frame. Seed replay also requires equal batch size,
    software versions and compatible hardware.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(p) or not 0 <= p <= 0.1:
        raise ValueError("p must be finite and between 0 and 0.1")
    return _normalize_result(_native_run_batch("zero", PROTOCOL_ID, p, n, seed))


def _normalize_result(data):
    """Normalize native arrays and verify terminal score; no physical evolution."""
    n = len(data["accepted"])
    data["completed"] = data["accepted"].copy()
    data["aborted"] = np.zeros(n, dtype=bool)
    data["status"] = 2 * data["accepted"].astype(np.uint8)
    data["resource_names"] = np.asarray(RESOURCE_NAMES)
    done = data["completed"]
    expected = terminal_oracle(data["Eout_x"], data["Eout_z"])
    if not np.array_equal(data["target_fail"][done], expected[done]):
        raise AssertionError("Physical terminal measurement disagrees with independent SteaneCode scoring")
    data["endpoint_radius_fail"] = done & endpoint_outside_radius_one(data["Eout_x"], data["Eout_z"])
    data["recovery_inclusive_fail"] = data["target_fail"] | data["endpoint_radius_fail"]
    data["historical_fail"] = data["target_fail"]
    return data


# ---------------------------------------------------------------------------
# Experiment: run this file directly to sample and draw its own curve.
# Only Python's standard library, NumPy, and Stim are required.
# ---------------------------------------------------------------------------
import argparse
import csv
from datetime import datetime
import hashlib
from html import escape
import math
from pathlib import Path
import platform
import sys
import time

DEFAULT_P = (0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007,
             0.001, 0.002, 0.003, 0.005, 0.007, 0.01)


def batch_seed(master, p, index):
    """Same seed partition as the verified package; not a paired experiment."""
    message = f"steane-qec-v1|{master}|{SCHEME}|{float(p).hex()}|{index}"
    return int.from_bytes(hashlib.sha256(message.encode("ascii")).digest()[:8], "big")


def wilson_interval(k, n):
    if not n:
        return None, None
    z = 1.959963984540054
    fraction = k / n
    denominator = 1 + z * z / n
    center = (fraction + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_point(p, total, metric):
    row = {"scheme": SCHEME, "label": LABEL, "p": p, **total}
    completed = total["completed"]
    row["completion_rate"] = completed / total["attempts"]
    for key in ("target", "endpoint_radius", "recovery_inclusive", "historical"):
        count = total[key + "_failures"]
        row[key + "_rate"] = count / completed if completed else None
        row[key + "_ci_low"], row[key + "_ci_high"] = wilson_interval(count, completed)
    row["failure_rate"] = row[metric + "_rate"]
    row["lower"], row["upper"] = row[metric + "_ci_low"], row[metric + "_ci_high"]
    row["curve_metric"] = {"target": "L", "endpoint_radius": "B", "recovery_inclusive": "L_or_B"}[metric]
    return row


def plot_curve(rows, output):
    """Draw a log-log SVG; zero estimates are upper-bound markers, not zeros.

    A downward triangle denotes a point with zero observed failures and is
    placed at its pointwise 95% Wilson upper bound. No positive rate is invented.
    Points with no completed outputs have undefined conditional rates and are
    omitted. If only p=0 was requested, use a linear x axis for that diagnostic.
    """
    valid = [row for row in rows if row["failure_rate"] is not None]
    positive = [row for row in valid if row["p"] > 0]
    data = sorted(positive or valid, key=lambda row: row["p"])
    width, height = 920, 600
    left, right, top, bottom = 100, 880, 90, 510
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" fill="#17263b" font-size="14">',
             f'<text x="100" y="31" font-size="23">{escape(LABEL)}: logical vs physical error</text>',
             f'<text x="100" y="55">Score {escape(rows[0]["curve_metric"])}; completed-output denominator; pointwise 95% Wilson intervals</text>']
    if not data:
        parts += ['<text x="100" y="160">No completed outputs: conditional logical error rate is undefined.</text>', '</g></svg>']
        output.write_text("\n".join(parts), encoding="utf-8")
        return
    log_x = bool(positive)
    xvalue = (lambda p: math.log10(p)) if log_x else (lambda p: p)
    xmin, xmax = min(xvalue(row["p"]) for row in data), max(xvalue(row["p"]) for row in data)
    if xmin == xmax:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    values = [float(row[key]) for row in data for key in ("failure_rate", "lower", "upper") if row[key] and row[key] > 0]
    ymin = math.floor(math.log10(min(values)))
    ymax = math.ceil(math.log10(max(values)))
    if ymin == ymax:
        ymin -= 1
    tx = lambda p: left + (xvalue(p) - xmin) / (xmax - xmin) * (right - left)
    ty = lambda value: bottom - (math.log10(max(value, 10 ** ymin)) - ymin) / (ymax - ymin) * (bottom - top)
    for exponent in range(ymin, ymax + 1):
        y = ty(10 ** exponent)
        parts += [f'<path d="M {left} {y:.2f} H {right}" stroke="#e2e7ed"/>',
                  f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end">1e{exponent}</text>']
    for row in data:
        x = tx(row["p"])
        parts += [f'<path d="M {x:.2f} {top} V {bottom}" stroke="#eef1f5"/>',
                  f'<text transform="translate({x:.2f},{bottom+17}) rotate(40)">{row["p"]:g}</text>']
    previous = None
    for row in data:
        x, rate = tx(row["p"]), row["failure_rate"]
        if rate == 0:
            y = ty(row["upper"])
            parts.append(f'<polygon points="{x-5:.2f},{y-4:.2f} {x+5:.2f},{y-4:.2f} {x:.2f},{y+5:.2f}" fill="#1565c0"/>')
            previous = None
            continue
        y = ty(rate)
        if previous is not None:
            parts.append(f'<path d="M {previous[0]:.2f} {previous[1]:.2f} L {x:.2f} {y:.2f}" stroke="#1565c0" stroke-width="2"/>')
        lo, hi = ty(row["lower"]), ty(row["upper"])
        parts += [f'<path d="M {x:.2f} {lo:.2f} V {hi:.2f} M {x-4:.2f} {lo:.2f} h 8 M {x-4:.2f} {hi:.2f} h 8" stroke="#1565c0"/>',
                  f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#1565c0"/>']
        previous = (x, y)
    parts += [f'<path d="M {left} {top} V {bottom} H {right}" fill="none" stroke="#17263b"/>',
              f'<text x="350" y="584">Physical error parameter p ({"log" if log_x else "linear"} scale)</text>',
              '<text transform="translate(28,405) rotate(-90)">Conditional failure probability (log scale)</text>',
              '<text x="100" y="77" fill="#526173">Downward triangle: zero observed failures, plotted at the 95% upper bound.</text>',
              '</g></svg>']
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=f"{LABEL}: standalone Steane QEC sampling and curve.")
    parser.add_argument("--p", type=float, nargs="+", default=list(DEFAULT_P))
    parser.add_argument("--shots", type=int, default=100000, help="Attempted inputs per p point")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--metric", choices=("target", "endpoint_radius", "recovery_inclusive"), default=DEFAULT_METRIC)
    parser.add_argument("--output", "--out", type=Path)
    parser.add_argument("--save-shots", action="store_true", help="Also retain raw per-shot arrays as NPZ")
    args = parser.parse_args()
    if args.shots < 1 or args.batch_size < 1 or any(not math.isfinite(p) or not 0 <= p <= .1 for p in args.p):
        parser.error("shots/batch-size must be positive and p must be finite in [0,0.1]")
    if len(set(args.p)) != len(args.p):
        parser.error("Duplicate p values are not allowed")
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent if script_dir.name == "programs" and (script_dir.parent / "requirements.txt").is_file() else script_dir
    output = args.output or project_dir / "output" / (Path(__file__).stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    information = [f"scheme={SCHEME}", f"metric={args.metric}", f"p={args.p}",
                   f"attempts_per_point={args.shots}", f"batch_size={args.batch_size}", f"master_seed={args.seed}",
                   f"stim={stim.__version__}", f"numpy={np.__version__}", f"python={sys.version}",
                   f"platform={platform.platform()}", f"source_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
                   "Input: one noisy verified logical-zero candidate per attempt.",
                   "Noise: reset10p,1q p,CX5p uniform15,report-flip2p,no idle,ideal virtual recovery.",
                   "L: logical-zero target failure AFTER additional ideal SteaneCode scoring.",
                   "B: output radius exceeds one BEFORE additional ideal SteaneCode scoring.",
                   "Denominator: completed QEC outputs; rejects and aborts are recorded separately.",
                   "Batch seeds use batch_seed(master,p,index). Keep batch partition, versions and CPU architecture for replay.",
                   "Intervals: pointwise 95% Wilson, not simultaneous coverage or a fault-tolerance proof."]
    (output / "run.txt").write_text("\n".join(information) + "\nstatus=RUNNING\n", encoding="utf-8")
    rows = []
    for point_index, p in enumerate(args.p):
        total = dict.fromkeys(("attempts", "input_accepted", "input_rejected", "qec_aborted", "completed",
                              "target_failures", "endpoint_radius_failures", "recovery_inclusive_failures", "historical_failures"), 0)
        started = time.perf_counter()
        for batch_index, start in enumerate(range(0, args.shots, args.batch_size)):
            count = min(args.batch_size, args.shots - start)
            data = run_batch(p, count, batch_seed(args.seed, p, batch_index))
            done = data["completed"]
            total["attempts"] += count
            total["input_accepted"] += int(data["accepted"].sum())
            total["input_rejected"] += int((~data["accepted"]).sum())
            total["qec_aborted"] += int(data["aborted"].sum())
            total["completed"] += int(done.sum())
            for metric in ("target", "endpoint_radius", "recovery_inclusive", "historical"):
                total[metric + "_failures"] += int(data[metric + "_fail"][done].sum())
            if args.save_shots:
                np.savez_compressed(output / f"p{point_index}_batch{batch_index}.npz",
                                    **{key: value for key, value in data.items() if isinstance(value, np.ndarray)})
        rows.append(summarize_point(p, total, args.metric))
        with (output / "curve.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{LABEL}: p={p:g}, complete={total['completed']}/{args.shots}, L={total['target_failures']}, B={total['endpoint_radius_failures']} ({time.perf_counter()-started:.2f}s)", flush=True)
    plot_curve(rows, output / "curve.svg")
    with (output / "run.txt").open("a", encoding="utf-8") as stream:
        stream.write("status=COMPLETE\n")
    print(f"Data: {(output / 'curve.csv').resolve()}\nFigure: {(output / 'curve.svg').resolve()}")


if __name__ == "__main__":
    main()

