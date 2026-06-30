"""Quantum phase estimation following arXiv:1910.11696 (Mohammadbagherpoor et al.).

Implements:
- Kitaev's algorithm (I / S ancilla prep + classical atan2 post-processing)
- Iterative QPE (Table I; Griffiths–Niu feed-forward) — single ancilla, dynamic
- Lloyd QPE (inverse QFT on n ancillas)
- Modified Lloyd / ACP: semi-classical IQFT (`if_test` replaces controlled-Rz)
- ACP iterative QPE (constant precision: only R2/R3 look-back)
"""

from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import PauliEvolutionGate, QFT, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter

PAPER_ARXIV = "1910.11696"


def resolve_trotter_reps(dt, trotter_reps=None):
    if trotter_reps is not None:
        return max(1, int(trotter_reps))
    return max(1, int(dt * 5))


def build_evolution_gate(
    hamiltonian: SparsePauliOp,
    tau: float,
    trotter_reps: int | None = None,
) -> Gate:
    reps = resolve_trotter_reps(tau, trotter_reps)
    return PauliEvolutionGate(
        hamiltonian,
        time=tau,
        synthesis=SuzukiTrotter(order=2, reps=reps),
    )


def build_phase_oracle_gate(phi: float) -> Gate:
    """U|1⟩ = exp(2πiφ)|1⟩ (paper Sec. III demo; UnitaryGate for correct `.power()`)."""
    p = float(phi)
    matrix = [[1.0, 0.0], [0.0, np.exp(2.0j * np.pi * p)]]
    return UnitaryGate(matrix, label=f"U({p:.4f})")


def prepare_neel(qc: QuantumCircuit, q_sys) -> None:
    for i, q in enumerate(q_sys):
        if i % 2 == 1:
            qc.x(q)


def prepare_eigenstate_one(qc: QuantumCircuit, q_sys) -> None:
    """|1⟩ on system qubit 0 (paper φ = 11/16 benchmark)."""
    qc.x(q_sys[0])


def bits_to_phase(bitstring_msb_first: str) -> float:
    """Map measured bits (MSB first) to phase φ in [0, 1)."""
    phi = 0.0
    for j, bit in enumerate(bitstring_msb_first):
        if bit == "1":
            phi += 1.0 / (2 ** (j + 1))
    return phi


def phase_to_energy(phi: float, tau: float, branch: int = 0) -> float:
    """Invert U = exp(-i H tau): E = 2π*(φ + branch) / τ."""
    return 2.0 * np.pi * (phi + float(branch)) / tau


def best_energy_branch(
    phi: float,
    tau: float,
    ref_energy: float,
    branches=range(-3, 4),
) -> dict:
    best = None
    for br in branches:
        e = phase_to_energy(phi, tau, branch=br)
        err = abs(e - ref_energy)
        if best is None or err < best["err"]:
            best = {"branch": br, "energy": e, "err": err}
    return best


# ---------------------------------------------------------------------------
# Feed-forward corrections (iterative / ACP)
# ---------------------------------------------------------------------------


def _apply_griffiths_niu_corrections(
    qc: QuantumCircuit, anc, c_bits, step: int
) -> None:
    """Full look-back — standard iterative QPE (paper Table I / Fig. 7)."""
    for prev in range(step):
        angle = -np.pi / (2 ** (step - prev))
        with qc.if_test((c_bits[prev], 1)):
            qc.p(angle, anc)


def _apply_acp_corrections(qc: QuantumCircuit, anc, c_bits, step: int) -> None:
    """ACP: only R2⁻¹ and R3⁻¹ from the two previous bits (paper Sec. II-D)."""
    if step >= 1:
        with qc.if_test((c_bits[step - 1], 1)):
            qc.p(-np.pi / 2, anc)
    if step >= 2:
        with qc.if_test((c_bits[step - 2], 1)):
            qc.p(-np.pi / 4, anc)


def _append_semclassical_iqft(
    qc: QuantumCircuit,
    q_anc,
    c_bits,
    n_bits: int,
) -> None:
    """
    Modified Lloyd / ACP IQFT (paper Fig. 11): deferred-measurement inverse QFT.

    Low-to-high: H → measure qubit ``m``; if outcome is 1, apply ``P(-π/2^{k-m})``
    on still-unmeasured qubits ``k > m``. Equivalent to ``QFT(n, inverse=True,
    do_swaps=False)`` (same schedule as inverse of ``qft_dynamic`` in
    ``dynamic_qft.ipynb``).
    """
    for m in range(n_bits):
        qc.h(q_anc[m])
        qc.measure(q_anc[m], c_bits[m])
        for k in range(m + 1, n_bits):
            angle = -np.pi / (2 ** (k - m))
            with qc.if_test((c_bits[m], 1)):
                qc.p(angle, q_anc[k])


# ---------------------------------------------------------------------------
# Iterative QPE (single ancilla, dynamic)
# ---------------------------------------------------------------------------


def build_dynamic_iqpe_circuit(
    hamiltonian: SparsePauliOp,
    tau: float,
    n_bits: int,
    n_sys: int,
    *,
    trotter_reps: int | None = None,
    reset_ancilla: bool = True,
    state_prep=prepare_neel,
) -> QuantumCircuit:
    """
    Iterative QPE (paper Table I, Fig. 7): one ancilla + MCM + Griffiths–Niu feed-forward.
    """
    u_gate = build_evolution_gate(hamiltonian, tau, trotter_reps)
    return _build_iterative_single_ancilla(
        u_gate,
        n_bits,
        n_sys,
        state_prep=state_prep,
        correction_fn=_apply_griffiths_niu_corrections,
        reset_ancilla=reset_ancilla,
        name="IQPE_iterative",
    )


def build_acp_iqpe_circuit(
    hamiltonian: SparsePauliOp,
    tau: float,
    n_bits: int,
    n_sys: int,
    *,
    trotter_reps: int | None = None,
    reset_ancilla: bool = True,
    state_prep=prepare_neel,
) -> QuantumCircuit:
    """ACP iterative QPE — 2-bit look-back only (paper Sec. II-D, Fig. 13–16)."""
    u_gate = build_evolution_gate(hamiltonian, tau, trotter_reps)
    return _build_iterative_single_ancilla(
        u_gate,
        n_bits,
        n_sys,
        state_prep=state_prep,
        correction_fn=_apply_acp_corrections,
        reset_ancilla=reset_ancilla,
        name="IQPE_acp",
    )


def _build_iterative_single_ancilla(
    u_gate: Gate,
    n_bits: int,
    n_sys: int,
    *,
    state_prep,
    correction_fn,
    reset_ancilla: bool,
    name: str,
) -> QuantumCircuit:
    q_anc = QuantumRegister(1, "anc")
    q_sys = QuantumRegister(n_sys, "sys")
    c_bits = ClassicalRegister(n_bits, "phase")
    qc = QuantumCircuit(q_anc, q_sys, c_bits, name=name)

    state_prep(qc, q_sys)

    for step in range(n_bits):
        power = 2 ** (n_bits - 1 - step)
        qc.h(q_anc[0])
        if step > 0:
            correction_fn(qc, q_anc[0], c_bits, step)
        qc.append(u_gate.power(power).control(1), [q_anc[0], *q_sys])
        qc.h(q_anc[0])
        qc.measure(q_anc[0], c_bits[step])
        if reset_ancilla and step < n_bits - 1:
            qc.reset(q_anc[0])

    return qc


# ---------------------------------------------------------------------------
# Lloyd QPE (n ancillas + inverse QFT)
# ---------------------------------------------------------------------------


def build_unitary_qpe_circuit(
    hamiltonian: SparsePauliOp,
    tau: float,
    n_bits: int,
    n_sys: int,
    *,
    trotter_reps: int | None = None,
    state_prep=prepare_neel,
) -> QuantumCircuit:
    """Lloyd QPE with unitary inverse QFT (paper Sec. II-C, Fig. 9)."""
    u_gate = build_evolution_gate(hamiltonian, tau, trotter_reps)
    return _build_lloyd_qpe(
        u_gate, n_bits, n_sys, state_prep=state_prep, semclassical=False
    )


def build_lloyd_semclassical_qpe_circuit(
    hamiltonian: SparsePauliOp,
    tau: float,
    n_bits: int,
    n_sys: int,
    *,
    trotter_reps: int | None = None,
    state_prep=prepare_neel,
) -> QuantumCircuit:
    """Modified Lloyd QPE — semi-classical IQFT (paper Fig. 11)."""
    u_gate = build_evolution_gate(hamiltonian, tau, trotter_reps)
    return _build_lloyd_qpe(
        u_gate, n_bits, n_sys, state_prep=state_prep, semclassical=True
    )


def _build_lloyd_qpe(
    u_gate: Gate,
    n_bits: int,
    n_sys: int,
    *,
    state_prep,
    semclassical: bool,
) -> QuantumCircuit:
    tag = "semclassical" if semclassical else "unitary"
    q_anc = QuantumRegister(n_bits, "anc")
    q_sys = QuantumRegister(n_sys, "sys")
    c_bits = ClassicalRegister(n_bits, "phase")
    qc = QuantumCircuit(q_anc, q_sys, c_bits, name=f"IQPE_lloyd_{tag}")

    state_prep(qc, q_sys)
    qc.h(q_anc)
    for j in range(n_bits):
        power = 2 ** (n_bits - 1 - j)
        qc.append(u_gate.power(power).control(1), [q_anc[j], *q_sys])

    if semclassical:
        _append_semclassical_iqft(qc, q_anc, c_bits, n_bits)
    else:
        qc.append(QFT(n_bits, inverse=True, do_swaps=False).to_gate(), q_anc)
        qc.measure(q_anc, c_bits)

    return qc


# ---------------------------------------------------------------------------
# Paper demo circuits (φ = 11/16, |1⟩)
# ---------------------------------------------------------------------------


def build_paper_demo_circuit(
    n_bits: int,
    *,
    method: str,
    phi: float,
) -> QuantumCircuit:
    """
    Phase-oracle QPE on eigenstate |1⟩ (demo helper for notebooks).

    method: ``kitaev_I``, ``kitaev_S``, ``iterative``, ``acp``,
            ``lloyd``, ``lloyd_semclassical``
    """
    u = build_phase_oracle_gate(phi)
    if method == "kitaev_I":
        return _build_kitaev_round(u, n_bits, k_op="I")
    if method == "kitaev_S":
        return _build_kitaev_round(u, n_bits, k_op="S")
    if method == "iterative":
        return _build_iterative_single_ancilla(
            u,
            n_bits,
            1,
            state_prep=prepare_eigenstate_one,
            correction_fn=_apply_griffiths_niu_corrections,
            reset_ancilla=True,
            name="paper_IQPE",
        )
    if method == "acp":
        return _build_iterative_single_ancilla(
            u,
            n_bits,
            1,
            state_prep=prepare_eigenstate_one,
            correction_fn=_apply_acp_corrections,
            reset_ancilla=True,
            name="paper_ACP",
        )
    if method == "lloyd":
        return _build_lloyd_qpe(
            u, n_bits, 1, state_prep=prepare_eigenstate_one, semclassical=False
        )
    if method == "lloyd_semclassical":
        return _build_lloyd_qpe(
            u, n_bits, 1, state_prep=prepare_eigenstate_one, semclassical=True
        )
    raise ValueError(f"Unknown method: {method}")


def _build_kitaev_round(u_gate: Gate, n_bits: int, *, k_op: str) -> QuantumCircuit:
    """Single Kitaev round at full precision power 2^{n-1} (paper Sec. II-A)."""
    q_anc = QuantumRegister(1, "anc")
    q_sys = QuantumRegister(1, "sys")
    c = ClassicalRegister(1, "m")
    qc = QuantumCircuit(q_anc, q_sys, c, name=f"Kitaev_{k_op}")

    prepare_eigenstate_one(qc, q_sys)
    qc.h(q_anc[0])
    if k_op == "S":
        qc.s(q_anc[0])
    power = 2 ** (n_bits - 1)
    qc.append(u_gate.power(power).control(1), [q_anc[0], q_sys[0]])
    qc.h(q_anc[0])
    qc.measure(q_anc[0], c[0])
    return qc


def kitaev_round_prob(counts: dict[str, int], outcome: str = "0") -> float:
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return counts.get(outcome, 0) / total


def kitaev_estimate_phi(
    counts_i: dict[str, int],
    counts_s: dict[str, int],
    n_bits: int,
) -> float:
    """
    Eq. (9): estimate φ from one Kitaev round at power 2^{n-1}.

    Returns φ mod 1 at n-bit resolution.
    """
    c_k = 2.0 * kitaev_round_prob(counts_i, "0") - 1.0
    s_k = 1.0 - 2.0 * kitaev_round_prob(counts_s, "0")
    phi_scaled = np.arctan2(s_k, c_k) / (2.0 * np.pi)
    if phi_scaled < 0:
        phi_scaled += 1.0
    return (phi_scaled / (2 ** (n_bits - 1))) % 1.0


def kitaev_round_counts(
    u_gate: Gate,
    power: int,
    k_op: str,
    shots: int,
    simulator=None,
) -> dict[str, int]:
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = simulator or AerSimulator()
    q_anc = QuantumRegister(1, "anc")
    q_sys = QuantumRegister(1, "sys")
    c = ClassicalRegister(1, "m")
    qc = QuantumCircuit(q_anc, q_sys, c)
    prepare_eigenstate_one(qc, q_sys)
    qc.h(q_anc[0])
    if k_op == "S":
        qc.s(q_anc[0])
    qc.append(u_gate.power(power).control(1), [q_anc[0], q_sys[0]])
    qc.h(q_anc[0])
    qc.measure(q_anc[0], c[0])
    return sim.run(transpile(qc, sim), shots=shots).result().get_counts()


def kitaev_demo_statistics(
    u_gate: Gate,
    n_bits: int,
    shots: int,
    simulator=None,
) -> dict:
    """
    Paper Sec. II-A: return P(0|I), P(0|S), Eq. (9) φ̂ and iterative-quality bits
    via the same controlled-U powers as Table I.
    """
    sim = simulator
    power = 2 ** (n_bits - 1)
    counts_i = kitaev_round_counts(u_gate, power, "I", shots, sim)
    counts_s = kitaev_round_counts(u_gate, power, "S", shots, sim)
    p0_i = kitaev_round_prob(counts_i, "0")
    p0_s = kitaev_round_prob(counts_s, "0")
    phi_hat = kitaev_estimate_phi(counts_i, counts_s, n_bits)

    # Accurate bits: reuse iterative circuit (paper IQPEA, Table I)
    qc = _build_iterative_single_ancilla(
        u_gate,
        n_bits,
        1,
        state_prep=prepare_eigenstate_one,
        correction_fn=_apply_griffiths_niu_corrections,
        reset_ancilla=True,
        name="kitaev_ref_iter",
    )
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    backend = sim or AerSimulator(method="statevector")
    ref_counts = backend.run(transpile(qc, backend), shots=shots).result().get_counts()
    _, _, bits = estimate_phase_from_counts(ref_counts, n_bits)

    return {
        "p0_I": p0_i,
        "p0_S": p0_s,
        "cos_est": 2 * p0_i - 1,
        "sin_est": 1 - 2 * p0_s,
        "phi_hat_coarse": phi_hat,
        "bits_iterative_ref": bits,
        "counts_I": counts_i,
        "counts_S": counts_s,
    }


# ---------------------------------------------------------------------------
# Count parsing
# ---------------------------------------------------------------------------


def normalize_phase_bitstring(raw: str, n_bits: int) -> str:
    """Keep the last ``n_bits`` of a count key (phase register)."""
    bits = raw.split()[0] if " " in raw else raw
    bits = bits.replace(" ", "")
    return bits[-n_bits:].zfill(n_bits)


def estimate_phase_from_counts(
    counts: dict[str, int],
    n_bits: int,
    *,
    msb_register_first: bool = True,
    reverse_bits: bool = False,
) -> tuple[float, float, str]:
    """
    Return (best_phase, probability, best_bitstring) from measurement counts.

    For all paper demo builders, ``phase`` bits are MSB-first in the count string
    (``c[0]`` … ``c[n-1]``). Set ``reverse_bits=True`` only when the backend
    returns reversed classical-register bit order.
    """
    best_bits = None
    best_prob = -1.0
    total = sum(counts.values())
    for key, count in counts.items():
        bits = normalize_phase_bitstring(key, n_bits)
        if len(bits) < n_bits:
            continue
        prob = count / total
        if prob > best_prob:
            best_prob = prob
            best_bits = bits
    if best_bits is None:
        return float("nan"), 0.0, ""
    if reverse_bits:
        best_bits = best_bits[::-1]
    elif not msb_register_first:
        best_bits = best_bits[::-1]
    return bits_to_phase(best_bits), best_prob, best_bits


def expected_bits(phi: float, n_bits: int) -> str:
    """Binary MSB-first string for φ at n_bits precision."""
    scaled = int(round(phi * (2**n_bits))) % (2**n_bits)
    return format(scaled, f"0{n_bits}b")
