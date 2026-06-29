"""Construct unitary and dynamic GHZ-state preparation circuits."""

from numbers import Integral

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr

__all__ = [
    "dynamic_ghz",
    "dynamic_ghz_circuit",
    "unitary_ghz",
    "unitary_ghz_circuit",
]


def _validate_num_qubits(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("n must be an integer")
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    return n


def _add_barrier(circuit: QuantumCircuit, enabled: bool) -> None:
    if enabled:
        circuit.barrier()


def unitary_ghz_circuit(
    n: int,
    *,
    do_measure: bool = True,
    add_barriers: bool = False,
) -> QuantumCircuit:
    """Return the canonical ``H + CX ladder`` GHZ circuit.

    The circuit has ``n - 1`` CNOT gates and two-qubit depth ``n - 1``.

    Args:
        n: Number of qubits in the GHZ state.
        do_measure: Add a terminal measurement when ``True``.
        add_barriers: Add visual barriers between preparation and measurement.
    """
    n = _validate_num_qubits(n)
    qubits = QuantumRegister(n, "q")
    circuit = QuantumCircuit(qubits, name="ghz_unitary")

    circuit.h(qubits[0])
    for control in range(n - 1):
        circuit.cx(qubits[control], qubits[control + 1])

    if do_measure:
        _add_barrier(circuit, add_barriers)
        final_bits = ClassicalRegister(n, "final_measurement")
        circuit.add_register(final_bits)
        circuit.measure(qubits, final_bits)

    return circuit


def dynamic_ghz_circuit(
    n: int,
    *,
    do_measure: bool = True,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Return the constant-depth dynamic GHZ circuit from Fig. 5.

    The construction works for both odd and even ``n``. Qubits
    ``1, 3, ...`` between neighboring even-indexed qubits are measured
    mid-circuit. Prefix XORs of those outcomes control Pauli-X corrections,
    after which the measured qubits are reset and re-entangled.

    For ``n >= 3``, the logical two-qubit gate depth is three.

    Args:
        n: Number of qubits in the GHZ state.
        do_measure: Add a terminal measurement when ``True``.
        add_barriers: Add visual barriers between the protocol stages.
    """
    n = _validate_num_qubits(n)
    qubits = QuantumRegister(n, "q")
    circuit = QuantumCircuit(qubits, name="ghz_dynamic")

    final_bits = None
    if do_measure:
        final_bits = ClassicalRegister(n, "final_measurement")
        circuit.add_register(final_bits)

    mid_qubits = list(range(1, n - 1, 2))
    mid_bits = None
    if mid_qubits:
        mid_bits = ClassicalRegister(len(mid_qubits), "mid_measurement")
        circuit.add_register(mid_bits)

    # Prepare |+> on q0, q2, ... .
    circuit.h(qubits[0:n:2])
    _add_barrier(circuit, add_barriers)

    # Two parallel CNOT layers encode the parity of adjacent even qubits.
    for right in range(2, n, 2):
        circuit.cx(qubits[right], qubits[right - 1])
    _add_barrier(circuit, add_barriers)

    for left in range(0, n - 2, 2):
        circuit.cx(qubits[left], qubits[left + 1])
    _add_barrier(circuit, add_barriers)

    if mid_bits is not None:
        for bit, qubit in zip(mid_bits, mid_qubits):
            circuit.measure(qubits[qubit], bit)
        _add_barrier(circuit, add_barriers)

        # Correct q2, q4, ... using cumulative measurement parity.
        parity = expr.lift(mid_bits[0])
        for bit_index, target in enumerate(range(2, n, 2)):
            if bit_index > 0:
                parity = expr.bit_xor(parity, mid_bits[bit_index])
            with circuit.if_test(parity):
                circuit.x(qubits[target])
        _add_barrier(circuit, add_barriers)

        for qubit in mid_qubits:
            circuit.reset(qubits[qubit])
        _add_barrier(circuit, add_barriers)

    # Restore the measured qubits and attach the last qubit when n is even.
    for control in range(0, n - 1, 2):
        circuit.cx(qubits[control], qubits[control + 1])

    if do_measure:
        _add_barrier(circuit, add_barriers)
        circuit.measure(qubits, final_bits)

    return circuit


# Short aliases retained for notebook readability and existing analysis cells.
dynamic_ghz = dynamic_ghz_circuit
unitary_ghz = unitary_ghz_circuit
