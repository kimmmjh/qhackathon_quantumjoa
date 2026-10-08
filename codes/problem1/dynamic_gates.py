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


def _counts_total_shots(counts: dict) -> int:
    return int(sum(int(value) for value in counts.values())) if counts else 0


def _count_key_to_bits(count_key, total_clbits: int = 0) -> str:
    """Normalize Qiskit count keys, including spaced registers and hex keys."""
    clean = str(count_key).replace(" ", "")
    if clean.startswith("0x"):
        bits = bin(int(clean, 16))[2:]
    else:
        bits = "".join(char for char in clean if char in "01")
    if total_clbits:
        bits = bits.zfill(total_clbits)
    return bits


def _counts_pauli_expectation(
    counts: dict,
    measured_clbits: list[int] | tuple[int, ...],
    total_clbits: int,
) -> float | None:
    """Return the product-Pauli expectation for measured classical bit indices."""
    if not measured_clbits:
        return 1.0

    shots = _counts_total_shots(counts)
    if shots == 0:
        return None

    total = 0.0
    for count_key, count in counts.items():
        bits = _count_key_to_bits(count_key, total_clbits)
        eigenvalue = 1
        for clbit in measured_clbits:
            bit = bits[-1 - clbit] if clbit < len(bits) else "0"
            eigenvalue *= -1 if bit == "1" else 1
        total += eigenvalue * int(count)
    return total / shots


def _expectation_from_counts(counts: dict, pauli: str) -> float:
    """Map final ``cr`` bitstrings to ⟨P⟩ for P in {XX, YY, ZZ}."""
    if pauli not in {"XX", "YY", "ZZ"}:
        raise ValueError(pauli)
    if not counts:
        return 0.0

    total_clbits = max(len(_count_key_to_bits(bitstring)) for bitstring in counts)
    expectation = _counts_pauli_expectation(counts, [0, 1], total_clbits)
    return 0.0 if expectation is None else expectation


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


CNOT_PROCESS_PAULI_TERMS = [
    {"term": "IIII", "ideal": 1},
    {"term": "IXIX", "ideal": 1},
    {"term": "IYZY", "ideal": 1},
    {"term": "IZZZ", "ideal": 1},
    {"term": "XIXX", "ideal": 1},
    {"term": "XXXI", "ideal": 1},
    {"term": "XYYZ", "ideal": 1},
    {"term": "XZYY", "ideal": -1},
    {"term": "YIYX", "ideal": 1},
    {"term": "YXYI", "ideal": 1},
    {"term": "YYXZ", "ideal": -1},
    {"term": "YZXY", "ideal": 1},
    {"term": "ZIZI", "ideal": 1},
    {"term": "ZXZX", "ideal": 1},
    {"term": "ZYIY", "ideal": 1},
    {"term": "ZZIZ", "ideal": 1},
]


def _local_pauli_prep_options(pauli: str) -> list[dict]:
    if pauli == "I":
        return [{"label": "0", "eigenvalue": 1}, {"label": "1", "eigenvalue": 1}]
    if pauli in {"X", "Y", "Z"}:
        return [{"label": "+", "eigenvalue": 1}, {"label": "-", "eigenvalue": -1}]
    raise ValueError(f"Unsupported Pauli {pauli!r}.")


def _prepare_local_pauli_eigenstate(qc: QuantumCircuit, qubit, pauli: str, option: dict) -> None:
    label = option["label"]
    eigenvalue = int(option["eigenvalue"])
    if pauli == "I":
        if label == "1":
            qc.x(qubit)
        return
    if pauli == "Z":
        if eigenvalue == -1:
            qc.x(qubit)
        return
    if eigenvalue == -1:
        qc.x(qubit)
    qc.h(qubit)
    if pauli == "Y":
        qc.s(qubit)


def _measure_local_pauli(qc: QuantumCircuit, qubit, pauli: str, clbit) -> None:
    if pauli == "I":
        return
    if pauli == "X":
        qc.h(qubit)
    elif pauli == "Y":
        qc.sdg(qubit)
        qc.h(qubit)
    elif pauli != "Z":
        raise ValueError(f"Unsupported Pauli {pauli!r}.")
    qc.measure(qubit, clbit)


def _compose_base_cnot_circuit(process_circuit: QuantumCircuit, base_circuit: QuantumCircuit) -> None:
    qubits = list(range(base_circuit.num_qubits))
    if base_circuit.num_clbits:
        process_circuit.compose(
            base_circuit,
            qubits=qubits,
            clbits=list(range(base_circuit.num_clbits)),
            inplace=True,
        )
    else:
        process_circuit.compose(base_circuit, qubits=qubits, inplace=True)


def build_cnot_process_circuits(circuit: QuantumCircuit) -> dict:
    """Build preparation/measurement circuits for the 16 CNOT process terms.

    The identity output term is known exactly, so only 15 terms x 4 input
    eigenstate preparations are executed.
    """
    if circuit.num_qubits < 2:
        raise ValueError("CNOT process certification requires at least two circuit qubits.")

    control = int((circuit.metadata or {}).get("control_index", 0))
    target = int((circuit.metadata or {}).get("target_index", circuit.num_qubits - 1))

    process_terms = []
    process_specs = []
    process_circuits = []

    for term_info in CNOT_PROCESS_PAULI_TERMS:
        term = term_info["term"]
        ideal = int(term_info["ideal"])
        input_pauli = term[:2]
        output_pauli = term[2:]
        process_terms.append(
            {
                "term": term,
                "ideal": ideal,
                "input_pauli": input_pauli,
                "output_pauli": output_pauli,
                "executed": output_pauli != "II",
            }
        )
        if output_pauli == "II":
            continue

        control_options = _local_pauli_prep_options(input_pauli[0])
        target_options = _local_pauli_prep_options(input_pauli[1])
        for control_option in control_options:
            for target_option in target_options:
                qc = QuantumCircuit(circuit.num_qubits, circuit.num_clbits, name=f"proc_{term}")
                _prepare_local_pauli_eigenstate(qc, control, input_pauli[0], control_option)
                _prepare_local_pauli_eigenstate(qc, target, input_pauli[1], target_option)
                _compose_base_cnot_circuit(qc, circuit)

                measured_qubits = [
                    qubit
                    for qubit, pauli in ((control, output_pauli[0]), (target, output_pauli[1]))
                    if pauli != "I"
                ]
                process_register = ClassicalRegister(len(measured_qubits), "proc")
                qc.add_register(process_register)

                measured_clbits = []
                measured_register_bits = []
                next_clbit = 0
                for qubit, pauli in ((control, output_pauli[0]), (target, output_pauli[1])):
                    if pauli == "I":
                        continue
                    _measure_local_pauli(qc, qubit, pauli, process_register[next_clbit])
                    measured_clbits.append(qc.find_bit(process_register[next_clbit]).index)
                    measured_register_bits.append(next_clbit)
                    next_clbit += 1

                prep_weight = int(control_option["eigenvalue"] * target_option["eigenvalue"])
                spec = {
                    "term": term,
                    "ideal": ideal,
                    "input_pauli": input_pauli,
                    "output_pauli": output_pauli,
                    "control_prep": dict(control_option),
                    "target_prep": dict(target_option),
                    "prep_weight": prep_weight,
                    "prep_average_weight": 0.25,
                    "measured_clbits": measured_clbits,
                    "num_clbits": qc.num_clbits,
                    "measurement_register": "proc",
                    "measured_register_bits": measured_register_bits,
                    "measurement_register_size": len(process_register),
                    "circuit_index": len(process_circuits),
                }
                qc.metadata = dict(circuit.metadata or {})
                qc.metadata.update({"process_certification": spec})
                process_specs.append(spec)
                process_circuits.append(qc)

    return {
        "terms": process_terms,
        "specs": process_specs,
        "circuits": process_circuits,
    }


def calculate_cnot_process_fidelity(
    process_specs: list[dict],
    counts_list: list[dict] | None,
    process_terms: list[dict] | None = None,
    counts_register: str | None = None,
) -> dict | None:
    """Estimate CNOT process and average gate fidelity from process-term counts."""
    if counts_list is None:
        return None

    process_terms = process_terms or CNOT_PROCESS_PAULI_TERMS
    term_estimates = {"IIII": 1.0}
    term_shots = {"IIII": None}
    partial_sums = {}
    partial_shots = {}

    for spec, counts in zip(process_specs, counts_list):
        if counts is None:
            return None
        if counts_register is not None and counts_register == spec.get("measurement_register"):
            measured_bits = spec.get("measured_register_bits", spec["measured_clbits"])
            num_bits = spec.get("measurement_register_size", spec["num_clbits"])
        else:
            measured_bits = spec["measured_clbits"]
            num_bits = spec["num_clbits"]

        expectation = _counts_pauli_expectation(counts, measured_bits, num_bits)
        if expectation is None:
            return None

        term = spec["term"]
        partial_sums[term] = (
            partial_sums.get(term, 0.0)
            + spec["prep_average_weight"] * spec["prep_weight"] * expectation
        )
        partial_shots[term] = partial_shots.get(term, 0) + _counts_total_shots(counts)

    for term, estimate in partial_sums.items():
        term_estimates[term] = estimate
        term_shots[term] = partial_shots.get(term)

    rows = []
    process_sum = 0.0
    for term_info in process_terms:
        term = term_info["term"]
        ideal = int(term_info["ideal"])
        estimate = term_estimates.get(term)
        if estimate is None:
            return None

        contribution = ideal * estimate / 16.0
        process_sum += contribution
        rows.append(
            {
                "term": term,
                "ideal": ideal,
                "estimate": estimate,
                "ideal_times_estimate": ideal * estimate,
                "process_contribution": contribution,
                "shots": term_shots.get(term),
            }
        )

    return {
        "process_fidelity": process_sum,
        "gate_fidelity": (4.0 * process_sum + 1.0) / 5.0,
        "term_rows": rows,
    }


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
