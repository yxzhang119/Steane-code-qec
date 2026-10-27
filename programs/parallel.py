"""Standalone Parallel Steane [[7,1,3]] QEC experiment.

Contains the full circuit templates, simulator, decoder and physical protocol.
Runtime dependencies: Python standard library, NumPy and Stim only.
No other local source, decoder table, JSON or NPZ file is required.

The workflow is noisy logical-zero preparation, bare logical-Z verification,
adaptive QEC, then ideal terminal SteaneCode recovery for scoring. The terminal
scoring operations are not part of the noisy QEC or its resource count.
"""
from __future__ import annotations

SCHEME = 'parallel'
LABEL = 'Parallel'
DEFAULT_METRIC = "target"
SOURCE_CORE_SHA256 = "84354cdc0c5adb55d9150fb5d9303b59b4ea41aaceee58f08a8c6ee10ddfb1e5"

import hashlib
import math
import itertools
from functools import lru_cache

import numpy as np
import stim

CANONICAL_GENERATORS = ((85, 0), (0, 102), (0, 120), (0, 85), (102, 0), (120, 0))

DATA_QUBITS = tuple(range(7))

STATE_BY_ID = {'zero': {'id': 'zero'}, 'one': {'id': 'one'}}

GENERATORS = {1: ('X', (0, 2, 4, 6)), 2: ('Z', (1, 2, 5, 6)), 3: ('Z', (3, 4, 5, 6)), 4: ('Z', (0, 2, 4, 6)), 5: ('X', (1, 2, 5, 6)), 6: ('X', (3, 4, 5, 6))}

SELECTED_GENERATORS = {'A': {'000': (), '001': (3, 4), '010': (2, 4), '011': (2, 4), '100': (1, 5, 6), '101': (1, 2, 3, 4, 5, 6), '110': (1, 2, 3, 4, 5, 6), '111': (1, 2, 3, 4, 5, 6)}, 'B': {'000': (), '001': (1, 6), '010': (1, 5), '011': (1, 5), '100': (2, 3, 4), '101': (1, 2, 3, 4, 5, 6), '110': (1, 2, 3, 4, 5, 6), '111': (1, 2, 3, 4, 5, 6)}}

def syndrome(x, z):
    return tuple(((gx & z).bit_count() + (gz & x).bit_count() & 1 for gx, gz in CANONICAL_GENERATORS))

def pauli_masks(text):
    return (sum((1 << q for q, p in enumerate(text) if p in 'XY')), sum((1 << q for q, p in enumerate(text) if p in 'YZ')))

@lru_cache(maxsize=1)
def minimum_weight_partial():
    candidates = [((x | z).bit_count(), text, x, z)
                  for text in map(''.join, itertools.product('IXYZ', repeat=7))
                  for x, z in [pauli_masks(text)]]
    output = {}
    for selected in set(SELECTED_GENERATORS['A'].values()) | set(SELECTED_GENERATORS['B'].values()):
        table = {}
        for _, _, x, z in sorted(candidates):
            sigma = syndrome(x, z)
            table.setdefault(''.join(str(sigma[g - 1]) for g in selected), (x, z))
        output[selected] = table
    return output

@lru_cache(maxsize=1)
def decoder(policy='selective_radius'):
    if policy != 'selective_radius': raise ValueError(policy)
    source = {key: (x, z) for key, x, z in SOURCE_LOCAL_ROWS}
    local, completion = {'A': {}, 'B': {}}, minimum_weight_partial()
    for block, patterns in SELECTED_GENERATORS.items():
        for f, selected in patterns.items():
            for u, correction in completion[selected].items():
                key = f'{block}|{f}|{u}'
                local[block][key] = {'correction': source.get(key, correction), 'fallback': key not in source, 'source_key_known': key in source}
    return {'stage_A': {'entries': local['A']}, 'stage_B': {'local_fallback_entries': local['B'], 'entries': {key: {'correction': (x, z)} for key, x, z in HISTORY_ROWS}}}

@lru_cache(maxsize=1)
def local_maps(policy='selective_radius'):
    payload = decoder(policy)
    table = {b: np.zeros((512, 2), np.uint16) for b in ('A', 'B')}
    fallback = {b: np.zeros(512, np.uint8) for b in ('A', 'B')}
    for block, entries in [('A', payload['stage_A']['entries']), ('B', payload['stage_B']['local_fallback_entries'])]:
        for key, entry in entries.items():
            _, f, u = key.split('|')
            code = int(f, 2) * 64 + (int(u, 2) if u else 0)
            table[block][code], fallback[block][code] = entry['correction'], entry['fallback']
    bx, bz = (np.tile(table['B'][:, column], 512) for column in range(2))
    known = np.zeros(512 * 512, np.bool_)
    for code, x, z in HISTORY_CODES: bx[code], bz[code], known[code] = x, z, True
    return table, fallback, bx, bz, known

SOURCE_LOCAL_CODES = (
    ('A', 0, 0, 0), ('A', 64, 0, 0), ('A', 65, 4, 0), ('A', 67, 64, 0), ('A', 128, 0, 0), ('A', 129, 16, 0), ('A', 130, 32, 0), ('A', 131, 64, 0), ('A', 192, 17, 0), ('A', 193, 16, 0), ('A', 194, 32, 0),
    ('A', 195, 64, 0), ('A', 256, 0, 0), ('A', 257, 0, 17), ('A', 258, 0, 5), ('A', 259, 0, 32), ('A', 260, 0, 1), ('A', 261, 0, 16), ('A', 262, 0, 4), ('A', 263, 0, 64), ('A', 321, 0, 17), ('A', 323, 0, 32),
    ('A', 374, 4, 4), ('A', 383, 64, 64), ('A', 386, 0, 5), ('A', 387, 0, 32), ('A', 396, 16, 0), ('A', 412, 64, 0), ('A', 419, 0, 64), ('A', 429, 16, 16), ('A', 456, 17, 0), ('A', 460, 16, 0), ('A', 511, 64, 64),
    ('B', 0, 0, 0), ('B', 64, 0, 0), ('B', 66, 0, 4), ('B', 67, 0, 64), ('B', 128, 0, 0), ('B', 129, 0, 32), ('B', 130, 0, 16), ('B', 131, 0, 64), ('B', 192, 0, 17), ('B', 193, 0, 32), ('B', 194, 0, 16),
    ('B', 195, 0, 64), ('B', 256, 0, 0), ('B', 257, 1, 0), ('B', 258, 17, 0), ('B', 259, 16, 0), ('B', 260, 5, 0), ('B', 261, 4, 0), ('B', 262, 32, 0), ('B', 263, 64, 0), ('B', 328, 17, 0), ('B', 344, 32, 0),
    ('B', 374, 4, 4), ('B', 383, 64, 64), ('B', 400, 5, 0), ('B', 408, 32, 0), ('B', 412, 64, 0), ('B', 417, 0, 16), ('B', 419, 0, 64), ('B', 429, 16, 16), ('B', 449, 0, 17), ('B', 481, 0, 16), ('B', 511, 64, 64),
)

HISTORY_CODES = (
    (0, 0, 0), (64, 0, 0), (66, 0, 4), (67, 0, 64), (128, 0, 0), (129, 0, 32), (130, 0, 16), (131, 0, 64), (192, 0, 17), (193, 0, 32), (194, 0, 16), (195, 0, 64), (256, 0, 0), (257, 1, 0), (258, 17, 0),
    (259, 16, 0), (260, 5, 0), (261, 4, 0), (262, 32, 0), (263, 64, 0), (320, 0, 0), (328, 17, 0), (330, 34, 2), (344, 32, 0), (346, 32, 2), (348, 64, 0), (354, 0, 4), (363, 68, 64), (374, 4, 4), (382, 64, 4),
    (383, 64, 64), (384, 0, 0), (400, 5, 0), (401, 40, 8), (408, 32, 0), (409, 32, 8), (412, 64, 0), (417, 0, 16), (419, 0, 64), (429, 16, 16), (433, 80, 16), (439, 4, 64), (441, 32, 16), (443, 32, 64), (445, 64, 16),
    (449, 0, 17), (453, 1, 17), (469, 4, 68), (479, 64, 32), (481, 0, 16), (485, 1, 16), (505, 32, 16), (511, 64, 64), (32768, 2, 0), (32832, 2, 2), (32834, 0, 4), (32961, 0, 32), (33280, 0, 0), (33346, 0, 4), (34304, 0, 0),
    (34370, 0, 4), (65536, 8, 0), (65664, 8, 8), (65666, 0, 16), (65729, 0, 32), (66048, 0, 0), (66178, 0, 16), (66241, 0, 32), (66560, 0, 0), (66753, 0, 32), (67072, 0, 0), (67138, 0, 4), (67265, 0, 32), (98304, 0, 0), (98370, 0, 4),
    (98816, 0, 0), (99009, 0, 32), (99328, 0, 0), (99521, 0, 32), (99840, 0, 0), (131072, 0, 0), (131329, 1, 0), (131335, 64, 0), (131584, 0, 0), (131847, 64, 0), (132096, 0, 0), (132355, 16, 0), (132608, 0, 0), (132871, 64, 0), (133120, 0, 0),
    (133377, 1, 0), (133632, 0, 0), (134144, 0, 0), (134656, 0, 0), (134915, 16, 0), (134919, 64, 0), (163840, 0, 0), (164352, 0, 0), (165376, 0, 0), (172544, 52, 52), (173568, 52, 0), (174080, 0, 0), (178688, 0, 52), (181760, 0, 0), (191488, 0, 0),
    (196096, 0, 0), (196608, 0, 0), (197632, 0, 0), (198144, 0, 0), (201728, 0, 0), (202240, 0, 0), (202752, 0, 0), (203776, 0, 0), (204288, 0, 0), (210944, 0, 0), (212480, 0, 0), (214528, 0, 0), (219648, 0, 0), (220672, 0, 0), (228352, 0, 52),
    (233472, 0, 0), (235520, 0, 0), (237056, 0, 0), (249856, 52, 52), (250880, 52, 52), (251904, 0, 52), (261632, 0, 0),
)

def local_key(block, code):
    flag, u = format(code // 64, '03b'), code % 64
    size = len(SELECTED_GENERATORS[block][flag])
    return f"{block}|{flag}|{format(u, f'0{size}b') if size else ''}"

SOURCE_LOCAL_ROWS = tuple((local_key(block, code), x, z) for block, code, x, z in SOURCE_LOCAL_CODES)

HISTORY_ROWS = tuple((local_key('A', h // 512) + '||' + local_key('B', h % 512), x, z) for h, x, z in HISTORY_CODES)

PROTOCOL_ID = 'selective_radius'

PROTOCOLS = (PROTOCOL_ID,)

DISPLAY_NAME = DATA_NAME = 'parallel'

DEFAULT_STATE, DEFAULT_P, DEFAULT_ATTEMPTS = 'zero', 0.001, 10000

MASTER_SEED = DEFAULT_SEED = 2026101323

NOISE_MODEL = {'post_reset': 10, 'post_1q': 1, 'post_2q': 5, 'report_flip': 2, 'idle': 0}

ROUTE_A_CX = ((1, 0), (3, 5), (2, 6), (1, 4), (2, 0), (3, 6), (1, 5), (6, 4))

LOGICAL_Z_SUPPORT = (0, 5, 6)

PAULI_BITS = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}

PAULI1 = ('X', 'Y', 'Z')

PAULI2 = tuple((a + b for a, b in itertools.product('IXYZ', repeat=2) if a + b != 'II'))

RATES, SYNDROME_ANCILLA, FLAG_ANCILLA = (1, 2, 5, 10), 7, 8

RESOURCE_NAMES = ('CX_total', 'reset_total', 'measurement_total', 'qec_generators', 'QEC_CX', 'QEC_reset', 'QEC_measurement', 'primitive_total')

FROZEN_DECODER_JSON_SHA256 = 'aba6adde43afafd967a68ae2a353bca4db077b8462b4f6d7c9871f8f5e4f4b34'

def location_stage(location_id):
    return 'qec' if not location_id.startswith('prep.') else 'verification' if 'bareZ' in location_id else 'suffix' if '.suffix.' in location_id else 'encoding'

A_CX = ((7, 4), (6, 8), (5, 9), (7, 9), (7, 0), (4, 8), (1, 9), (7, 2), (3, 8), (6, 9), (7, 8), (7, 6), (5, 8), (2, 9))

B_CX = tuple(((t, c) for c, t in A_CX))

BLOCK_PREPARATIONS = {'A': (('RX', 7), ('R', 8), ('R', 9)), 'B': (('R', 7), ('RX', 8), ('RX', 9))}

BLOCK_READOUTS = {'A': (('MX', 7), ('M', 8), ('M', 9)), 'B': (('M', 7), ('MX', 8), ('MX', 9))}

@lru_cache(maxsize=256)
def gadget(kind, *args):
    ops, generators = [], 1
    def add(name, targets, location):
        ops.append((name, tuple(targets), location))
    if kind == 'prep':
        state, = args
        if state not in STATE_BY_ID: raise ValueError(state)
        add('R', (7,), 'prep.routeA.bareZ.R.ancilla')
        for q in DATA_QUBITS: add('R', (q,), f'prep.routeA.R.q{q}')
        for q in (1, 2, 3): add('H', (q,), f'prep.routeA.H.q{q}')
        for i, targets in enumerate(ROUTE_A_CX, 1): add('CX', targets, f'prep.routeA.CX.{i:02d}')
        for i, q in enumerate(LOGICAL_Z_SUPPORT, 1): add('CX', (q, 7), f'prep.routeA.bareZ.CX.{i:02d}')
        add('M', (7,), 'prep.routeA.bareZ.M.ancilla')
    elif kind == 'block':
        block, = args
        if block not in BLOCK_PREPARATIONS: raise ValueError(f'Unknown parallel block: {block!r}')
        generators = 3
        for i, (gate, q) in enumerate(BLOCK_PREPARATIONS[block], 1): add(gate, (q,), f'par.{block}.{gate}.{i:02d}')
        for i, targets in enumerate(A_CX if block == 'A' else B_CX, 1): add('CX', targets, f'par.{block}.CX.{i:02d}')
        for i, (gate, q) in enumerate(BLOCK_READOUTS[block], 1): add(gate, (q,), f'par.{block}.{gate}.{i:02d}')
    elif kind == 'bare':
        g, prefix = args
        axis, support = GENERATORS[g]
        prepare, measure = ('RX', 'MX') if axis == 'X' else ('R', 'M')
        add(prepare, (7,), f'{prefix}.{prepare}.syndrome')
        for i, q in enumerate(support, 1): add('CX', (7, q) if axis == 'X' else (q, 7), f'{prefix}.CX.{i:02d}')
        add(measure, (7,), f'{prefix}.{measure}.syndrome')
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

def followup(engine, idx, block, f, extra):
    local = f.astype(np.uint16) * 64
    for pattern, selected in SELECTED_GENERATORS[block].items():
        active = np.flatnonzero(f == int(pattern, 2))
        subset = idx[active]
        if not len(subset): continue
        u = np.zeros(len(subset), np.uint16)
        for g in selected:
            u = u * 2 + engine.execute('bare', subset, g, f'sel.{block}.followup.g{g}')[0]
        local[active] += u
        if extra and selected:
            for g in range(1, 7):
                if g not in selected: engine.execute('bare', subset, g, f'sel.{block}.extra.g{g}')
    return local

def _native_run_batch(state, protocol, p, n, seed, forced=None, record_events=False):
    if not 0 <= p <= 0.1 or n < 0: raise ValueError('p must lie in [0, 0.1] and n must be nonnegative')
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
    policy = PROTOCOL_ID
    maps, fb, bx, bz, known = local_maps(policy)
    extra = False
    if len(idx):
        fA = binary(engine.execute('block', idx, 'A'))
        a = followup(engine, idx, 'A', fA, extra)
        c = maps['A'][a]
        engine.frame(idx, c)
        frame_x[idx] ^= c[:, 0]
        frame_z[idx] ^= c[:, 1]
        fallback_A[idx] = fb['A'][a]
        fB = binary(engine.execute('block', idx, 'B'))
        b = followup(engine, idx, 'B', fB, extra)
        h = a.astype(np.int32) * 512 + b
        c = np.column_stack((bx[h], bz[h]))
        engine.frame(idx, c)
        frame_x[idx] ^= c[:, 0]
        frame_z[idx] ^= c[:, 1]
        fallback_B[idx] = ~known[h]
        history_code[idx] = h
    out_x, out_z = engine.frame_masks()
    failed = np.zeros(n, np.bool_)
    readout, terminal_syndrome = engine.terminal_readout(all_idx[accepted])
    failed[accepted] = readout
    logical_readout = np.full(n, -1, np.int8)
    logical_readout[accepted] = readout
    if forced is not None and (not np.all(engine.applied == forced.get('expected_count', 2))): raise AssertionError('A forced proposal was not reached exactly')
    output = {'accepted': accepted, 'target_fail': failed, 'logical_readout': logical_readout, 'Eout_x': out_x & 127, 'Eout_z': out_z & 127, 'preparation_x': prep_x, 'preparation_z': prep_z, 'frame_x': frame_x, 'frame_z': frame_z, 'resources': engine.resources, 'fallback_A': fallback_A, 'fallback_B': fallback_B, 'history_code': history_code, 'event_count': engine.event_count, 'prefix_survival_counts': engine.prefix}
    if record_events: output.update(events=engine.events, measurements=engine.measurements)
    return output

def depth_from_history(accepted, history_code):
    accepted = np.asarray(accepted, dtype=np.bool_)
    h = np.asarray(history_code, dtype=np.int64)
    if np.any(h[~accepted] != -1): raise AssertionError('Rejected preparations must have no QEC history')
    a = np.where(accepted, h // 512, 0)
    b = np.where(accepted, h % 512, 0)
    counts_A = np.array([len(SELECTED_GENERATORS['A'][format(f, '03b')]) for f in range(8)])
    counts_B = np.array([len(SELECTED_GENERATORS['B'][format(f, '03b')]) for f in range(8)])
    checks = counts_A[a // 64] + counts_B[b // 64]
    qec_depth = np.where(accepted, 12 + 4 * checks, 0)
    return {'QEC_CNOT_depth': qec_depth, 'CNOT_depth_total': qec_depth + 6}


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

