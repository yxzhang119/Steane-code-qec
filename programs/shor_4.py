"""Shor-4: independent Stim experiment implementation.

One noisy logical-zero candidate is verified once, followed by this file's
QEC controller. Recovery is a noiseless Pauli frame. Ideal terminal SteaneCode is
used only for scoring; output frames are saved before it.

The historical Steane-1 score is L OR B; the other schemes use L. L is the
fixed-|0_L> target failure, and B is output distance greater than one to any
codeword. Neither score alone proves fault tolerance for arbitrary inputs.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math

import numpy as np
import stim

SCHEME = 'shor_confirm_four'
LABEL = 'Shor-4'
DEFAULT_METRIC = 'target'



# Physical noise and persistent Stim frame backend.


NUM_QUBITS = 15


RESOURCE_NAMES = (
    "cx_total", "reset_total", "measurement_total", "h_total",
    "cx_init", "cx_qec", "primitive_total", "cx_depth_total",
    "cx_depth_init", "cx_depth_qec", "reset_init", "reset_qec",
    "measurement_init", "measurement_qec", "h_init", "h_qec",
    "oneq_total",
)


RESOURCE_INDEX = {name: i for i, name in enumerate(RESOURCE_NAMES)}


NOISE_MODEL = {
    "post_reset": 10, "post_1q": 1, "post_2q": 5,
    "report_flip": 2, "idle": 0,
}


_SCORING_SYNDROMES = stim.Circuit("""
MPP X0*X2*X4*X6 Z1*Z2*Z5*Z6 Z3*Z4*Z5*Z6 Z0*Z2*Z4*Z6 X1*X2*X5*X6 X3*X4*X5*X6
""")


_SCORING_LOGICAL_Z = stim.Circuit("MPP Z0*Z5*Z6")


@lru_cache(maxsize=512)
def _compile(text: str, p: float):
    """Cache a noisy fragment and its physical resource counts; never sample."""
    noisy = stim.Circuit()
    counts = np.zeros(len(RESOURCE_NAMES), dtype=np.uint32)
    depth = np.zeros(NUM_QUBITS, dtype=np.uint32)

    def add(name, value):
        counts[RESOURCE_INDEX[name]] += value

    for op in stim.Circuit(text).flattened():
        name = op.name
        if name in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        if op.gate_args_copy():
            raise ValueError("execute expects a noiseless physical fragment")
        targets = op.targets_copy()
        if any(not target.is_qubit_target for target in targets):
            raise ValueError("Physical fragments may only use qubit targets")
        qubits = [target.value for target in targets]
        if any(q < 0 or q >= NUM_QUBITS for q in qubits):
            raise ValueError("The backend has qubits q0 through q14")

        if name == "CX":
            if len(qubits) % 2:
                raise ValueError("CX targets must be pairs")
            add("cx_total", len(qubits) // 2)
            add("primitive_total", len(qubits) // 2)
            for a, b in zip(qubits[::2], qubits[1::2]):
                layer = max(int(depth[a]), int(depth[b])) + 1
                depth[a] = depth[b] = layer
                # Stim merges adjacent CX instructions, including pairs that
                # share a wire. Noise must stay immediately after each pair.
                noisy.append("CX", [a, b])
                noisy.append("DEPOLARIZE2", [a, b], 5 * p)
        elif name in ("R", "RX", "RY"):
            add("reset_total", len(qubits))
            add("primitive_total", len(qubits))
            noisy.append(name, qubits)
            noisy.append("DEPOLARIZE1", qubits, 10 * p)
        elif name in ("M", "MX", "MY"):
            add("measurement_total", len(qubits))
            add("primitive_total", len(qubits))
            noisy.append(name, qubits, 2 * p)
        elif name in ("H", "X", "Y", "Z", "S", "S_DAG"):
            add("oneq_total", len(qubits))
            add("h_total", len(qubits) if name == "H" else 0)
            add("primitive_total", len(qubits))
            if name in ("X", "Y", "Z"):
                # A deterministic physical Pauli changes our fixed reference
                # frame. FlipSimulator.do(X) alone only conjugates its frame.
                noisy.append(f"{name}_ERROR", qubits, 1.0)
            else:
                noisy.append(name, qubits)
            noisy.append("DEPOLARIZE1", qubits, p)
        else:
            raise ValueError(f"Unsupported physical instruction: {name}")

    # Shared-wire CX dependencies define layers; TICK is only a drawing marker.
    # Distinct execute() fragments are serialized by adding their depths.
    counts[RESOURCE_INDEX["cx_depth_total"]] = int(depth.max())
    return noisy, counts


class Engine:
    """Batched persistent error frames, with independent RNG streams per group.

    idx is a one-dimensional array of distinct shot indices. Group changes copy
    only Pauli frames; this is valid because all protocol decisions use ideal
    deterministic measurement products. No data state is reinitialized during
    a group change. Resets happen only where the supplied fragment requests R.
    """

    def __init__(self, n: int, p: float, seed: int):
        if int(n) != n or n <= 0:
            raise ValueError("n must be a positive integer")
        if not np.isfinite(p) or not 0 <= p <= 0.1:
            raise ValueError("p must be between 0 and 0.1")
        self.n, self.p, self.seed = int(n), float(p), int(seed)
        self.seed_counter = 0
        self.all_idx = np.arange(self.n, dtype=np.int64)
        self.sim = self._new_sim(self.n)
        self.active_sim, self.active_idx = self.sim, self.all_idx
        self.resources = np.zeros((self.n, len(RESOURCE_NAMES)), np.uint32)

    def _new_sim(self, size):
        digest = hashlib.sha256(
            f"four-scheme|{self.seed}|{self.seed_counter}".encode("ascii")
        ).digest()
        self.seed_counter += 1
        return stim.FlipSimulator(
            batch_size=size, num_qubits=NUM_QUBITS,
            disable_stabilizer_randomization=True,
            seed=int.from_bytes(digest[:8], "big"),
        )

    def _indices(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= self.n):
            raise ValueError("idx must contain valid shot indices")
        if len(idx) > 1 and np.any(idx[1:] <= idx[:-1]):
            if len(np.unique(idx)) != len(idx):
                raise ValueError("idx must not contain duplicate shots")
        return idx

    def _commit(self):
        if self.active_sim is self.sim:
            return
        old_x, old_z, *_ = self.sim.to_numpy(output_xs=True, output_zs=True)
        new_x, new_z, *_ = self.active_sim.to_numpy(output_xs=True, output_zs=True)
        for pauli, old, new in (("X", old_x, new_x), ("Z", old_z, new_z)):
            mask = np.zeros((NUM_QUBITS, self.n), dtype=np.bool_)
            mask[:, self.active_idx] = old[:, self.active_idx] ^ new
            self.sim.broadcast_pauli_errors(pauli=pauli, mask=mask)
        self.active_sim, self.active_idx = self.sim, self.all_idx

    def _activate(self, idx):
        if np.array_equal(idx, self.active_idx):
            return self.active_sim
        self._commit()
        if np.array_equal(idx, self.all_idx):
            return self.sim
        xs, zs, *_ = self.sim.to_numpy(output_xs=True, output_zs=True)
        child = self._new_sim(len(idx))
        child.broadcast_pauli_errors(pauli="X", mask=np.ascontiguousarray(xs[:, idx]))
        child.broadcast_pauli_errors(pauli="Z", mask=np.ascontiguousarray(zs[:, idx]))
        self.active_sim, self.active_idx = child, idx.copy()
        return child

    def execute(self, circuit, idx, stage="qec"):
        """Execute a physical fragment; return its measurement-flip arrays.

        stage is init/qec (or init.* / qec.*). Counts include all actual retry
        fragments. A whole fragment is one native Stim call, with cached noise.
        """
        idx = self._indices(idx)
        if stage == "init" or stage.startswith("init."):
            category = "init"
        elif stage == "qec" or stage.startswith("qec."):
            category = "qec"
        else:
            raise ValueError("stage must classify the fragment as init or qec")
        noisy, base_counts = _compile(str(circuit), self.p)
        count = int(base_counts[RESOURCE_INDEX["measurement_total"]])
        if not len(idx):
            return [np.empty(0, dtype=np.uint8) for _ in range(count)]
        delta = base_counts.copy()
        for total, partial in (
            ("cx_total", "cx"), ("reset_total", "reset"),
            ("measurement_total", "measurement"), ("h_total", "h"),
            ("cx_depth_total", "cx_depth"),
        ):
            delta[RESOURCE_INDEX[f"{partial}_{category}"]] = delta[RESOURCE_INDEX[total]]
        self.resources[idx] += delta
        sim = self._activate(idx)
        sim.do(noisy)
        return [
            sim.get_measurement_flips(record_index=i - count).astype(np.uint8)
            for i in range(count)
        ]

    def frame(self, idx, corrections):
        """Apply noiseless virtual X/Z frames, with no physical resource cost."""
        idx = self._indices(idx)
        corrections = np.asarray(corrections, dtype=np.uint16)
        if corrections.shape != (len(idx), 2):
            raise ValueError("corrections must have shape (len(idx), 2)")
        if np.any(corrections >= 1 << NUM_QUBITS):
            raise ValueError("Correction masks must fit the fifteen qubits")
        if not len(idx):
            return
        sim = self._activate(idx)
        shifts = np.arange(NUM_QUBITS, dtype=np.uint16)
        for column, pauli in enumerate(("X", "Z")):
            mask = ((corrections[:, column, None] >> shifts) & 1).T.astype(np.bool_)
            sim.broadcast_pauli_errors(pauli=pauli, mask=mask)

    def frame_masks(self):
        """Return all fifteen-qubit X/Z masks, including inactive groups."""
        self._commit()
        xs, zs, *_ = self.sim.to_numpy(output_xs=True, output_zs=True)
        powers = (np.uint16(1) << np.arange(NUM_QUBITS, dtype=np.uint16))[:, None]
        return tuple(
            (bits * powers).sum(axis=0, dtype=np.uint32).astype(np.uint16)
            for bits in (xs, zs)
        )

    def terminal_readout(self, idx):
        """Ideal SteaneCode recovery then logical-Z056 measurement; no resource cost.

        This scoring step mutates the selected frames. Its bool result measures
        logical X/Y target failure of |0L>, not arbitrary-state logical infidelity.
        """
        idx = self._indices(idx)
        if not len(idx):
            return np.empty(0, dtype=np.bool_)
        sim = self._activate(idx)
        sim.do(_SCORING_SYNDROMES)
        s = np.column_stack([
            sim.get_measurement_flips(record_index=i - 6).astype(np.uint16)
            for i in range(6)
        ])
        qx = s[:, 3] + 2 * s[:, 1] + 4 * s[:, 2]
        qz = s[:, 0] + 2 * s[:, 4] + 4 * s[:, 5]
        correction = np.column_stack([
            np.where(q, np.uint16(1) << (np.maximum(q, 1) - 1), 0)
            for q in (qx, qz)
        ]).astype(np.uint16)
        self.frame(idx, correction)
        sim.do(_SCORING_LOGICAL_Z)
        return sim.get_measurement_flips(record_index=-1).copy()


# Code convention, input preparation, and this scheme's extraction gadget.


DATA = list(range(7))


SUPPORTS = ([0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6])


GENERATORS = (("X", SUPPORTS[0]), ("Z", SUPPORTS[1]),
              ("Z", SUPPORTS[2]), ("Z", SUPPORTS[0]),
              ("X", SUPPORTS[1]), ("X", SUPPORTS[2]))


MAX_ANCILLA_ATTEMPTS = 2


SHOR_CAT = [7, 8, 9, 10]


MAX_ROUNDS = 4


zero_candidate = stim.Circuit("""
R 0 1 2 3 4 5 6 7
H 1 2 3
TICK
CX 1 0 3 5
TICK
CX 2 6
TICK
CX 1 4
TICK
CX 2 0 3 6
TICK
CX 1 5 6 4
TICK
CX 0 7
TICK
CX 5 7
TICK
CX 6 7
TICK
M 7
TICK
""")


cat_checked = stim.Circuit("""
R 7 8 9 10 11
H 7
TICK
CX 7 8
TICK
CX 8 9
TICK
CX 9 10
TICK
CX 7 11
TICK
CX 10 11
TICK
M 11
TICK
""")


def shor_readout(axis, support):
    circuit = stim.Circuit()
    if axis == "Z":
        circuit.append("H", SHOR_CAT)
        circuit.append("TICK")
    for a, q in zip(SHOR_CAT, support):
        circuit.append("CX", [a, q] if axis == "X" else [q, a])
    circuit.append("TICK")
    circuit.append("MX" if axis == "X" else "M", SHOR_CAT)
    circuit.append("TICK")
    return circuit


# Classical syndrome decoding and adaptive controller.


def correction_masks(syndrome):
    syndrome = np.asarray(syndrome, dtype=np.uint16)
    x_label = syndrome[:, 3] + 2 * syndrome[:, 1] + 4 * syndrome[:, 2]
    z_label = syndrome[:, 0] + 2 * syndrome[:, 4] + 4 * syndrome[:, 5]
    masks = np.zeros((len(syndrome), 2), dtype=np.uint16)
    for column, labels in enumerate((x_label, z_label)):
        nonzero = labels != 0
        masks[nonzero, column] = np.left_shift(
            np.uint16(1), labels[nonzero] - np.uint16(1)
        )
    return masks


def _new_result(engine):
    return {
        "completed": np.empty(0, dtype=np.int64),
        "aborted": np.empty(0, dtype=np.int64),
        "rounds": np.zeros(engine.n, dtype=np.uint8),
        "attempted_rounds": np.zeros(engine.n, dtype=np.uint8),
        "ancilla_rejects": np.zeros(engine.n, dtype=np.uint16),
        "ancilla_attempts": np.zeros(engine.n, dtype=np.uint16),
        "abort_reason": np.zeros(engine.n, dtype=np.uint8),
        "syndrome": np.zeros((engine.n, 6), dtype=np.uint8),
    }


def _checked_ancilla(engine, idx, circuit, stage, result):
    pending = np.asarray(idx, dtype=np.int64)
    accepted_parts = []
    for _ in range(MAX_ANCILLA_ATTEMPTS):
        if not len(pending):
            break
        checks = np.asarray(engine.execute(circuit, pending, stage=stage)[0], dtype=np.uint8)
        result["ancilla_attempts"][pending] += 1
        result["ancilla_rejects"][pending] += checks
        accepted_parts.append(pending[checks == 0])
        pending = pending[checks != 0]
    if len(pending):
        result["abort_reason"][pending] = 1
    accepted = np.concatenate(accepted_parts) if accepted_parts else np.empty(0, dtype=np.int64)
    return np.sort(accepted)


def _parity(bits):
    return np.bitwise_xor.reduce(np.asarray(bits, dtype=np.uint8), axis=0)


def _finish_round(engine, active, syndrome, previous, round_index, result, fixed_two=False):
    result["rounds"][active] += 1
    result["syndrome"][active] = syndrome[active]
    if fixed_two:
        result["syndrome_first" if round_index == 0 else "syndrome_last"][active] = syndrome[active]
        if round_index == 1:
            result["syndrome_disagree"][active] = np.any(syndrome[active] != previous[active], axis=1)
        # Record the first round only; recover from the second-round syndrome even if it differs.
        confirmed = np.full(len(active), round_index == 1, dtype=bool)
    else:
        confirmed = np.all(syndrome[active] == previous[active], axis=1) if round_index else np.zeros(len(active), dtype=bool)
    done = active[confirmed]
    if len(done):
        engine.frame(done, correction_masks(syndrome[done]))
    remaining = active[~confirmed]
    previous[remaining] = syndrome[remaining]
    return done, remaining


def run_shor(engine, idx, stage="qec", fixed_two=False):
    result = _new_result(engine)
    if fixed_two:
        result["syndrome_first"] = np.zeros((engine.n, 6), dtype=np.uint8)
        result["syndrome_last"] = np.zeros((engine.n, 6), dtype=np.uint8)
        result["syndrome_disagree"] = np.zeros(engine.n, dtype=np.bool_)
    initial = np.asarray(idx, dtype=np.int64)
    active = initial.copy()
    previous = np.full((engine.n, 6), 255, dtype=np.uint8)
    completed = []
    for round_index in range(2 if fixed_two else MAX_ROUNDS):
        if not len(active):
            break
        result["attempted_rounds"][active] += 1
        syndrome = np.zeros((engine.n, 6), dtype=np.uint8)
        for column, (axis, support) in enumerate(GENERATORS):
            active = _checked_ancilla(engine, active, cat_checked, stage, result)
            if not len(active):
                break
            bits = engine.execute(shor_readout(axis, support), active, stage=stage)
            syndrome[active, column] = _parity(bits)
        if not len(active):
            break
        done, active = _finish_round(engine, active, syndrome, previous, round_index, result, fixed_two)
        completed.append(done)
    result["abort_reason"][active] = 2
    result["completed"] = np.sort(np.concatenate(completed)) if completed else np.empty(0, dtype=np.int64)
    result["aborted"] = initial[result["abort_reason"][initial] != 0]
    return result


def run_qec(engine, idx):
    """Apply Shor-4 to existing data; do not prepare or score it."""
    return run_shor(engine, idx, stage="qec")


# Resource-depth convention; no physical gates are changed here.


def depth_with_verification_barrier(circuit, verifier):
    """Serialize the completed encoder and its verification, as in Step22.

    Within each phase use shared-wire CNOT dependency layers. H/reset/measurement
    times and TICK markers are excluded. This is a resource metric, not latency.
    """
    layers = [np.zeros(15, dtype=int), np.zeros(15, dtype=int)]
    phase = 0
    for op in circuit:
        if op.name != "CX":
            continue
        targets = [t.value for t in op.targets_copy()]
        for a, b in zip(targets[::2], targets[1::2]):
            if verifier in (a, b):
                phase = 1
            elif phase:
                raise ValueError("Verification must follow the completed encoder")
            layer = max(layers[phase][a], layers[phase][b]) + 1
            layers[phase][a] = layers[phase][b] = layer
    return sum(int(part.max()) for part in layers)


DEPTH_MODELS = {}


for _circuit, _verifier in ((zero_candidate, 7), (cat_checked, 11)):
    _raw_depth = int(_compile(str(_circuit), 0)[1][RESOURCE_INDEX["cx_depth_total"]])
    _aligned_depth = depth_with_verification_barrier(_circuit, _verifier)
    DEPTH_MODELS[str(_circuit)] = {"original_fragment_depth": _raw_depth,
                                  "phase_barrier_depth": _aligned_depth,
                                  "delta": _aligned_depth - _raw_depth}


class DepthAlignedEngine(Engine):
    """Preserve every physical operation; align only the reported depth metric."""
    def __init__(self, n, p, seed):
        super().__init__(n, p, seed)
        self.depth_alignment = np.zeros((n, 2), np.uint16)  # init, QEC

    def execute(self, circuit, idx, stage="qec"):
        output = super().execute(circuit, idx, stage)
        delta = DEPTH_MODELS.get(str(circuit), {}).get("delta", 0)
        if delta and len(idx):
            category = "init" if stage.startswith("init") else "qec"
            self.resources[idx, RESOURCE_INDEX["cx_depth_total"]] += delta
            self.resources[idx, RESOURCE_INDEX["cx_depth_" + category]] += delta
            self.depth_alignment[idx, 0 if category == "init" else 1] += delta
        return output


def terminal_oracle(x, z):
    """Independent fixed Hamming-SteaneCode calculation, with low-bit qubit ordering."""
    del z  # The prepared-zero logical-Z target only depends on X components.
    x = np.asarray(x, dtype=np.uint16)
    syndrome = [np.bitwise_count(x & mask) & 1 for mask in (85, 102, 120)]
    address = syndrome[0] + 2 * syndrome[1] + 4 * syndrome[2]
    correction = np.where(address, np.uint16(1) << (np.maximum(address, 1) - 1), 0)
    return (np.bitwise_count((x ^ correction) & 97) & 1).astype(bool)


def endpoint_outside_radius_one(x, z):
    """Distance of the output Pauli to ANY valid codeword exceeds one.

    For this Hamming parity check, the minimum Pauli weight in a syndrome
    class exceeds one exactly when its X and Z labels are distinct and both
    nonzero.  This criterion is not the error distance to the input state and
    is not an event-wise fault-tolerance test at the actual internal fault count.
    """
    x = np.asarray(x, dtype=np.uint16)
    z = np.asarray(z, dtype=np.uint16)

    def labels(mask):
        bits = [np.bitwise_count(mask & support) & 1 for support in (85, 102, 120)]
        return bits[0] | (bits[1] << 1) | (bits[2] << 2)

    a, b = labels(x), labels(z)
    return (a != 0) & (b != 0) & (a != b)


# One complete physical batch and separate scoring definitions.


def run_batch(p: float, n: int, seed: int):
    """Run ``n`` attempted preparations, returning raw arrays and status.

    ``accepted`` marks initialization acceptance; ``completed`` marks QEC
    completion, and ``aborted`` marks accepted inputs that aborted inside QEC.
    Status is 0=rejected input, 1=QEC abort, 2=QEC complete.  Failure booleans
    are only meaningful on completed shots and are false placeholders elsewhere.
    The same scheme, environment, seed AND batch partition are needed for
    deterministic replay. Different schemes are not paired random experiments.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(p) or not 0 <= p <= 0.1:
        raise ValueError("p must be finite and between 0 and 0.1")
    engine = DepthAlignedEngine(n, p, seed)
    all_indices = np.arange(engine.n, dtype=np.int64)
    verification = engine.execute(zero_candidate, all_indices, stage="init")[0]
    accepted = verification == 0
    initial_resources = engine.resources.copy()
    preparation_x, preparation_z = engine.frame_masks()
    preparation_x &= 127
    preparation_z &= 127
    result = run_qec(engine, all_indices[accepted])
    done = result["completed"]
    completed = np.zeros(engine.n, dtype=bool)
    completed[done] = True
    status = accepted.astype(np.uint8)
    status[done] = 2
    out_x, out_z = engine.frame_masks()
    out_x &= 127
    out_z &= 127
    resources = engine.resources.copy()
    target_fail = np.zeros(engine.n, dtype=bool)
    if len(done):
        target_fail[done] = engine.terminal_readout(done)
        if not np.array_equal(target_fail[done], terminal_oracle(out_x[done], out_z[done])):
            raise AssertionError("Stim terminal scoring disagrees with the independent SteaneCode oracle")
    if not np.array_equal(resources, engine.resources):
        raise AssertionError("Terminal scoring changed physical resource counts")
    endpoint_fail = completed & endpoint_outside_radius_one(out_x, out_z)
    data = {
        "status": status,
        "accepted": accepted,
        "completed": completed,
        "aborted": accepted & ~completed,
        "target_fail": target_fail,
        "endpoint_radius_fail": endpoint_fail,
        "recovery_inclusive_fail": target_fail | endpoint_fail,
        "preparation_verification": verification,
        "preparation_x": preparation_x,
        "preparation_z": preparation_z,
        "Eout_x": out_x,
        "Eout_z": out_z,
        "resources": resources,
        "init_resources": initial_resources,
        "depth_alignment": engine.depth_alignment,
        "resource_names": np.asarray(RESOURCE_NAMES),
    }
    for key, values in result.items():
        if isinstance(values, np.ndarray) and values.shape[:1] == (engine.n,):
            data["qec_" + key] = values
    data["historical_fail"] = (data["recovery_inclusive_fail"]
                               if DEFAULT_METRIC == "recovery_inclusive" else data["target_fail"])
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

