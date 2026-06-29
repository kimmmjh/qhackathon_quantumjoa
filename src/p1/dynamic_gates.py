"""
Dynamic long-range entanglement circuits (Qiskit 2.x).

Reference: Bäumer et al., PRX Quantum 5, 030339 (arXiv:2308.13065)
- Appendix A / Fig. 4: dynamic long-range CNOT (no H inside the gate)
- Appendix B / Fig. 6: unitary baseline Strategy Ic (4n+1 CNOT)
- Appendix A.2 / Fig. 5: dynamic GHZ (MQT Bench ghz_dynamic pattern)
- Appendix C: Bell-state fidelity F = (1 + ⟨XX⟩ - ⟨YY⟩ + ⟨ZZ⟩) / 4
"""

from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr


def _check_even(n: int) -> int:
    """Bell-pair chain start index: 1 if n ancillas is even, else 2."""
    return 1 if n % 2 == 0 else 2


def build_dynamic_cnot(n_ancilla: int) -> QuantumCircuit:
    """
    Constant-depth long-range CNOT via gate teleportation (Fig. 4).

    Layout on a 1D chain: ``sys[0]`` (control) — ``anc[0..n-1]`` — ``sys[1]`` (target).
    Ancillas start in |0⟩; they are sacrificial (measured, not long-term memory).

    Args:
        n_ancilla: Number of ancilla qubits between the two system qubits (paper ``n``).

    Returns:
        Dynamic circuit without state preparation on control/target.
        Classical register ``cr`` (2 bits) is reserved for optional final readout.
    """
    if n_ancilla < 0:
        raise ValueError("n_ancilla must be non-negative.")

    n = n_ancilla
    k = n // 2
    # Physical 1D order: control — anc[0..n-1] — target (Fig. 4).
    control_reg = QuantumRegister(1, "control")
    target_reg = QuantumRegister(1, "target")
    anc = QuantumRegister(n, "anc") if n else QuantumRegister(0, "anc")
    cr = ClassicalRegister(2, "cr")
    cregs: list = [cr]
    if n > 1:
        cregs.append(ClassicalRegister(k, "c_zz"))
    if n > 0:
        cregs.append(ClassicalRegister(n - k, "c_xx"))

    qc = QuantumCircuit(control_reg, anc, target_reg, *cregs)
    q = [control_reg[0]] + list(anc) + [target_reg[0]]
    control, target = q[0], q[-1]

    x0 = _check_even(n)
    if n % 2 != 0:
        qc.cx(control, q[1])

    for i in range(k):
        qc.h(q[x0 + 2 * i])
        qc.cx(q[x0 + 2 * i], q[x0 + 2 * i + 1])

    for i in range(k + 1):
        qc.cx(q[x0 - 1 + 2 * i], q[x0 + 2 * i])

    for i in range(1, k + x0):
        qc.h(q[2 * i + 1 - x0])

    if n > 1:
        c_zz = qc.cregs[1]
        for i in range(k):
            qc.measure(q[2 * i + x0], c_zz[i])

    if n > 0:
        c_xx = qc.cregs[-1]
        for i in range(1, k + x0):
            qc.measure(q[2 * i + 1 - x0], c_xx[i - 1])

    if n > 0:
        c_xx = qc.cregs[-1]
        parity_xx = expr.lift(c_xx[0])
        for i in range(2, k + x0):
            parity_xx = expr.bit_xor(c_xx[i - 1], parity_xx)
        with qc.if_test(parity_xx):
            qc.z(control)

    if n > 1:
        c_zz = qc.cregs[1]
        parity_zz = expr.lift(c_zz[0])
        for i in range(1, k):
            parity_zz = expr.bit_xor(c_zz[i], parity_zz)
        with qc.if_test(parity_zz):
            qc.x(target)

    return qc


def build_unitary_cnot_ic(n_ancilla: int) -> QuantumCircuit:
    """
    Ancilla-bus unitary CNOT, Strategy Ic (Fig. 6): 4n+1 CNOT, tidle ≈ 0.

    Same layout as ``build_dynamic_cnot``: sys[0] — anc — sys[1].
    """
    if n_ancilla < 0:
        raise ValueError("n_ancilla must be non-negative.")

    n = n_ancilla
    control_reg = QuantumRegister(1, "control")
    target_reg = QuantumRegister(1, "target")
    anc = QuantumRegister(n, "anc") if n else QuantumRegister(0, "anc")
    cr = ClassicalRegister(2, "cr")
    qc = QuantumCircuit(control_reg, anc, target_reg, cr, name="CNOT_unitary_Ic")

    q = [control_reg[0]] + list(anc) + [target_reg[0]]
    num = len(q)
    k = n // 2

    for i in range(k):
        qc.cx(q[i], q[i + 1])
        qc.cx(q[i + 1], q[i])
        qc.cx(q[num - 1 - i], q[num - 2 - i])
        qc.cx(q[num - 2 - i], q[num - 1 - i])

    if n % 2 == 1:
        qc.cx(q[k + 2], q[k + 1])
        qc.cx(q[k + 1], q[k + 2])

    qc.cx(q[k], q[k + 1])

    for i in range(k):
        qc.cx(q[k - i], q[k - 1 - i])
        qc.cx(q[k - 1 - i], q[k - i])
        qc.cx(q[k + i + 1], q[k + i + 2])
        qc.cx(q[k + i + 2], q[k + i + 1])

    if n % 2 == 1:
        qc.cx(q[num - 2], q[num - 1])
        qc.cx(q[num - 1], q[num - 2])

    return qc


def _control_target_qubits(qc: QuantumCircuit) -> tuple:
    """Return (control, target) qubits: first and last quantum registers."""
    return qc.qregs[0][0], qc.qregs[-1][0]


def measure_cnot_in_basis(qc: QuantumCircuit, basis: str = "XX") -> QuantumCircuit:
    """
    Append Pauli-basis measurement of control and target onto ``cr``.

    basis must be one of ``XX``, ``YY``, ``ZZ`` (Appendix C.2).
    """
    if basis not in {"XX", "YY", "ZZ"}:
        raise ValueError("basis must be 'XX', 'YY', or 'ZZ'")

    out = qc.copy()
    control, target = _control_target_qubits(out)
    cr = out.cregs[0]

    if basis == "XX":
        out.h(control)
        out.h(target)
    elif basis == "YY":
        out.sdg(control)
        out.sdg(target)
        out.h(control)
        out.h(target)

    out.measure(control, cr[0])
    out.measure(target, cr[1])
    return out


def build_cnot_bell_test(n_ancilla: int, basis: str = "XX") -> QuantumCircuit:
    """Prepare |+⟩|0⟩, apply dynamic CNOT, measure in ``basis``."""
    body = build_dynamic_cnot(n_ancilla)
    control, _ = _control_target_qubits(body)
    prep = QuantumCircuit(*body.qregs, *body.cregs)
    prep.h(control)
    qc = prep.compose(body, inplace=False)
    return measure_cnot_in_basis(qc, basis=basis)


def build_unitary_cnot_bell_test(n_ancilla: int, basis: str = "XX") -> QuantumCircuit:
    """Prepare |+⟩|0⟩, apply Strategy-Ic unitary CNOT, measure in ``basis``."""
    body = build_unitary_cnot_ic(n_ancilla)
    control, _ = _control_target_qubits(body)
    prep = QuantumCircuit(*body.qregs, *body.cregs)
    prep.h(control)
    qc = prep.compose(body, inplace=False)
    return measure_cnot_in_basis(qc, basis=basis)


def _expectation_from_counts(counts: dict, pauli: str) -> float:
    """Map Z-basis bitstrings to ⟨P⟩ for P in {XX, YY, ZZ} on (control, target)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    exp = 0.0
    for bitstring, freq in counts.items():
        bits = bitstring.replace(" ", "")[-2:]
        c0, c1 = int(bits[0]), int(bits[1])
        if pauli == "ZZ":
            sign = 1 if c0 == c1 else -1
        elif pauli in {"XX", "YY"}:
            sign = 1 if c0 != c1 else -1
        else:
            raise ValueError(pauli)
        exp += sign * freq
    return exp / total


def cnot_fidelity_from_counts(
    counts_xx: dict,
    counts_yy: dict,
    counts_zz: dict,
) -> float:
    """Bell-state fidelity (Appendix C.2): F = (1 + ⟨XX⟩ - ⟨YY⟩ + ⟨ZZ⟩) / 4."""
    xx = _expectation_from_counts(counts_xx, "XX")
    yy = _expectation_from_counts(counts_yy, "YY")
    zz = _expectation_from_counts(counts_zz, "ZZ")
    return 0.25 * (1.0 + xx - yy + zz)


def bell_correlation_from_counts(counts: dict) -> float:
    """Fraction of shots with correlated outcomes in {'00', '11'} on ``cr``."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    good = sum(
        v
        for k, v in counts.items()
        if k.replace(" ", "")[-2:] in {"00", "11"}
    )
    return good / total


def mid_measurement_qubits(n_data: int) -> list[int]:
    """Odd chain qubits measured mid-circuit (paper Appendix A.2: n/2 - 1)."""
    return list(range(1, n_data - 1, 2))


def build_dynamic_ghz(n_data: int, final_measure: bool = False) -> QuantumCircuit:
    """
    Constant-depth GHZ on ``n_data`` qubits (Fig. 5 / arXiv:2308.13065).

    Paper resource scaling: ~3n/2−1 CNOT, n/2−1 mid-circuit measurements, O(1) depth.
    Mid-circuit: measure ``q1 … q_{n-2}``, prefix-XOR feed-forward, then ``reset``.

    Args:
        n_data: Number of data qubits on the 1D chain.
        final_measure: If True, measure all qubits into ``final_measurement``.
    """
    if n_data < 1:
        raise ValueError("n_data must be at least 1.")

    q = QuantumRegister(n_data, "q")
    if final_measure:
        final_cr = ClassicalRegister(n_data, "final_measurement")
        qc = QuantumCircuit(q, final_cr, name="ghz_dynamic")
    else:
        qc = QuantumCircuit(q, name="ghz_dynamic")

    if n_data == 1:
        qc.h(q[0])
        if final_measure:
            qc.measure(q, qc.cregs[0])
        return qc

    mid_qubits = mid_measurement_qubits(n_data)
    if mid_qubits:
        qc.add_register(ClassicalRegister(len(mid_qubits), "mid_measurement"))

    qc.h(range(0, n_data, 2))

    for right in range(2, n_data, 2):
        qc.cx(right, right - 1)
    for left in range(0, n_data - 1, 2):
        if left + 1 < n_data - 1:
            qc.cx(left, left + 1)

    if mid_qubits:
        mid = qc.cregs[-1]
        for clbit, qubit in zip(mid, mid_qubits):
            qc.measure(qubit, clbit)

        parity = expr.lift(mid[0])
        for bit, target in enumerate(range(2, n_data, 2)):
            if bit > 0:
                parity = expr.bit_xor(parity, mid[bit])
            with qc.if_test(parity):
                qc.x(target)

        for qubit in mid_qubits:
            qc.reset(qubit)

    for i in range(0, n_data - 1, 2):
        qc.cx(i, i + 1)

    if final_measure:
        qc.measure(q, qc.cregs[0])

    return qc


def build_unitary_ghz(n_data: int) -> QuantumCircuit:
    """Standard linear CNOT chain GHZ: H(0); CX chain — n−1 CNOT, O(n) depth."""
    if n_data < 1:
        raise ValueError("n_data must be at least 1.")
    q = QuantumRegister(n_data, "q")
    cr = ClassicalRegister(n_data, "final_measurement")
    qc = QuantumCircuit(q, cr, name="ghz_unitary")
    qc.h(q[0])
    for i in range(n_data - 1):
        qc.cx(q[i], q[i + 1])
    qc.measure(q, cr)
    return qc


def summarize_ghz_counts(counts: dict, n_data: int) -> dict:
    """Count all-|0⟩ and all-|1⟩ outcomes on the last ``n_data`` bits."""
    ok = 0
    for bitstring, freq in counts.items():
        bits = bitstring.replace(" ", "")[-n_data:]
        if bits in {"0" * n_data, "1" * n_data}:
            ok += freq
    total = sum(counts.values())
    return {"GHZ_Success": ok, "Error": total - ok}


def count_resources(qc: QuantumCircuit) -> dict:
    """Gate / measurement counts for Table I / II comparison."""
    ops = qc.count_ops()
    return {
        "cx": ops.get("cx", 0),
        "h": ops.get("h", 0),
        "measure": ops.get("measure", 0),
        "if_else": ops.get("if_else", 0),
    }


if __name__ == "__main__":
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    print("=== Dynamic CNOT (Bell test) ===")
    for n in [0, 1, 3, 5]:
        qc = build_cnot_bell_test(n, "XX")
        counts = sim.run(qc, shots=1024).result().get_counts()
        print(f"  n_anc={n}: Bell {100 * bell_correlation_from_counts(counts):.1f}%")

    print("=== Dynamic GHZ ===")
    for n in [4, 6, 8]:
        qc = build_dynamic_ghz(n, final_measure=True)
        counts = sim.run(qc, shots=1024).result().get_counts()
        s = summarize_ghz_counts(counts, n)
        print(
            f"  n={n}: GHZ {100 * s['GHZ_Success'] / 1024:.1f}% "
            f"resources={count_resources(qc)}"
        )
