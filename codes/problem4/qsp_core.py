import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter

TWO_QUBIT_GATE_NAMES = frozenset(
    {"cx", "cz", "cy", "ch", "cp", "ecr", "rzz", "rxx", "ryy", "rzx", "iswap"}
)
_TWO_QUBIT_FILTER = lambda inst: inst.operation.num_qubits == 2


def _resolve_trotter_reps(dt, trotter_reps=None):
    if trotter_reps is not None:
        return max(1, int(trotter_reps))
    return max(1, int(dt * 5))


def count_2q_gates(qc):
    """Count native / synthesized two-qubit gates (CX, ECR, CZ, ...)."""
    ops = qc.count_ops()
    return int(sum(ops.get(name, 0) for name in TWO_QUBIT_GATE_NAMES))


def profile_circuit(qc, label=None):
    """Return logical or ISA circuit resource metrics; print if label is set."""
    ops = qc.count_ops()
    stats = {
        "depth": qc.depth(),
        "depth_2q": qc.depth(_TWO_QUBIT_FILTER),
        "n_2q": count_2q_gates(qc),
        "n_mcm": int(ops.get("measure", 0)),
        "n_reset": int(ops.get("reset", 0)),
        "n_if_else": int(ops.get("if_else", 0)),
    }
    if label:
        print(
            f"{label}: depth={stats['depth']} | 2Q-depth={stats['depth_2q']} | "
            f"2Q-count={stats['n_2q']} | MCM={stats['n_mcm']} | reset={stats['n_reset']}"
        )
    return stats


def build_heisenberg_hamiltonian(N, J, h):
    """Creates a 1D Heisenberg XYZ Hamiltonian."""
    pauli_list = []
    coeffs = []
    Jx, Jy, Jz = J

    for i in range(N - 1):
        for op, coupling in zip(['X', 'Y', 'Z'], [Jx, Jy, Jz]):
            if coupling != 0.0:
                s = ['I'] * N
                s[N - 1 - i] = op
                s[N - 1 - (i + 1)] = op
                pauli_list.append("".join(s))
                coeffs.append(coupling)
                
    if h != 0.0:
        for i in range(N):
            s = ['I'] * N
            s[N - 1 - i] = 'Z'
            pauli_list.append("".join(s))
            coeffs.append(h)

    return SparsePauliOp(pauli_list, coeffs)

def exact_diagonalization(hamiltonian):
    """Performs exact diagonalization to find the true ground state energy."""
    matrix = hamiltonian.to_matrix()
    evals, evecs = np.linalg.eigh(matrix)
    return evals[0], evecs[:, 0]

def create_qsp_base_circuit(hamiltonian, dt, degree, N, trotter_reps=None):
    """Creates a pure QSP filter circuit without basis measurements."""
    q_anc = QuantumRegister(1, 'ancilla')
    q_sys = QuantumRegister(N, 'system')
    c_anc = ClassicalRegister(degree, 'c_anc') # For ancilla qubit measurement (d bits)
    c_sys = ClassicalRegister(N, 'c_sys')      # For system measurement (N bits)
    
    qc = QuantumCircuit(q_anc, q_sys, c_anc, c_sys)
    
    # Prepare N'eel State |0101>
    for i in range(N):
        if i % 2 == 1:
            qc.x(q_sys[i])
    qc.barrier()
    
    reps = _resolve_trotter_reps(dt, trotter_reps)
    trotter_gate = PauliEvolutionGate(hamiltonian, time=dt, synthesis=SuzukiTrotter(order=2, reps=reps))
    c_trotter = trotter_gate.control(1)
    
    # QSP Filtering Loop
    for k in range(degree):
        qc.h(q_anc[0])
        qc.append(c_trotter, [q_anc[0]] + list(q_sys))
        qc.h(q_anc[0])
        qc.measure(q_anc[0], c_anc[k])
        qc.reset(q_anc[0])
        qc.barrier()
        
    return qc, q_sys, c_sys

def add_basis_measurement(base_qc, q_sys, c_sys, basis):
    """Adds rotation gates and measurements corresponding to a specific Pauli basis (X, Y, Z) at the end of the circuit."""
    qc = base_qc.copy()
    if basis == 'X':
        qc.h(q_sys)
    elif basis == 'Y':
        qc.sdg(q_sys)
        qc.h(q_sys)
    # No rotation needed for Z basis
    qc.measure(q_sys, c_sys)
    return qc

def calculate_energy_from_shots(counts_dict, N, J, h):
    """
    Receives shot data for each basis (X, Y, Z), performs post-selection,
    and calculates the expected energy and 1-sigma standard error.
    """
    Jx, Jy, Jz = J
    expected_energy = 0.0
    total_variance = 0.0  # Variable to accumulate variance for error propagation
    passed_shots_info = {}

    for basis, counts in counts_dict.items():
        shot_energies = [] # Record individual shot energies for variance calculation
        valid_shots = 0
        
        for bitstr_combo, count in counts.items():
            sys_str, anc_str = bitstr_combo.split()
            
            # [Post-selection] Only shots where the ancilla qubit is '0' pass
            if '1' not in anc_str:
                valid_shots += count
                physical_bits = sys_str[::-1]
                
                shot_energy = 0.0
                for i in range(N - 1):
                    val = 1 if physical_bits[i] == physical_bits[i+1] else -1
                    if basis == 'X': shot_energy += Jx * val
                    elif basis == 'Y': shot_energy += Jy * val
                    elif basis == 'Z': shot_energy += Jz * val
                
                if basis == 'Z' and h != 0.0:
                    for i in range(N):
                        val = 1 if physical_bits[i] == '0' else -1
                        shot_energy += h * val
                        
                # Add shots with the same energy to the array 'count' times
                shot_energies.extend([shot_energy] * count)
                
        if valid_shots > 1:
            mean_E = np.mean(shot_energies)
            # Calculate sample variance (ddof=1)
            var_E = np.var(shot_energies, ddof=1) 
            expected_energy += mean_E
            # Error propagation: Add the variance of each basis mean (var/N)
            total_variance += var_E / valid_shots 
        elif valid_shots == 1:
            expected_energy += shot_energies[0]
            
        passed_shots_info[basis] = valid_shots

    total_passed = passed_shots_info.get('Z', 0)
    # Calculate final 1-sigma standard error
    standard_error = np.sqrt(total_variance)
    
    return expected_energy, standard_error, total_passed


def calculate_fidelity_theory(hamiltonian, dt, degree, N, target_sv, trotter_reps=None):
    """
    Calculates the theoretical fidelity for the 'success (Ancilla=0)' branch of the QSP filter circuit
    mathematically rigorously through direct manipulation of the Statevector.
    """
    q_anc = QuantumRegister(1, 'ancilla')
    q_sys = QuantumRegister(N, 'system')
    qc = QuantumCircuit(q_anc, q_sys)
    
    # Initial state |0101>
    for i in range(N):
        if i % 2 == 1: 
            qc.x(q_sys[i])
            
    current_sv = Statevector(qc)
    
    # 2nd order Trotter decomposition
    reps = _resolve_trotter_reps(dt, trotter_reps)
    trotter_gate = PauliEvolutionGate(hamiltonian, time=dt, synthesis=SuzukiTrotter(order=2, reps=reps))
    c_trotter = trotter_gate.control(1)
    
    # Single layer circuit
    layer_qc = QuantumCircuit(q_anc, q_sys)
    layer_qc.h(q_anc[0])
    layer_qc.append(c_trotter, [q_anc[0]] + list(q_sys))
    layer_qc.h(q_anc[0])
    
    for _ in range(degree):
        current_sv = current_sv.evolve(layer_qc)
        data = np.array(current_sv.data)
        
        # [Mathematical Projection Trick]
        # q_anc is index 0 (rightmost LSB).
        # Therefore, states where q_anc is 1 are located at 'odd indices' in the Statevector array.
        # To project to Ancilla=0, all odd indices are set to 0.
        data[1::2] = 0.0 
        
        # Renormalization
        norm = np.linalg.norm(data)
        if norm > 1e-10:
            data /= norm
        current_sv = Statevector(data)
        
    # Extract final system state: Collecting only 'even indices' where q_anc is 0 results in a pure system register state.
    final_sys_vec = current_sv.data[0::2] 
    
    fid = abs(np.vdot(target_sv, final_sys_vec))**2
    return fid

def calculate_ideal_metrics_statevector(hamiltonian, dt, degree, N, target_sv, trotter_reps=None):
    """
    After projecting by manipulating the Statevector,
    this function returns both the theoretical fidelity (Ideal Fidelity) and theoretical energy (Ideal Energy).
    """
    q_anc = QuantumRegister(1, 'ancilla')
    q_sys = QuantumRegister(N, 'system')
    qc = QuantumCircuit(q_anc, q_sys)
    
    for i in range(N):
        if i % 2 == 1: qc.x(q_sys[i])
            
    current_sv = Statevector(qc)
    
    reps = _resolve_trotter_reps(dt, trotter_reps)
    trotter_gate = PauliEvolutionGate(hamiltonian, time=dt, synthesis=SuzukiTrotter(order=2, reps=reps))
    c_trotter = trotter_gate.control(1)
    
    layer_qc = QuantumCircuit(q_anc, q_sys)
    layer_qc.h(q_anc[0])
    layer_qc.append(c_trotter, [q_anc[0]] + list(q_sys))
    layer_qc.h(q_anc[0])
    
    for _ in range(degree):
        current_sv = current_sv.evolve(layer_qc)
        data = np.array(current_sv.data)
        
        data[1::2] = 0.0 # Ancilla=0 Projection (remove odd indices)
        
        norm = np.linalg.norm(data)
        if norm > 1e-10: data /= norm
        current_sv = Statevector(data)
        
    final_sys_vec = current_sv.data[0::2] 
    
    # 1. Calculate Ideal Fidelity
    ideal_fid = abs(np.vdot(target_sv, final_sys_vec))**2
    
    # 2. Calculate Ideal Energy (using H matrix)
    H_mat = hamiltonian.to_matrix()
    ideal_energy = np.real(np.vdot(final_sys_vec, H_mat @ final_sys_vec))
    
    return ideal_fid, ideal_energy

def build_adiabatic_circuit(N, H_target, steps, total_time=10.0):
    """
    Creates a ground state preparation circuit based on Adiabatic Evolution.
    This serves as a control group for comparing performance against the proposed QSP filter in terms of circuit depth (Cost).
    """
    qr = QuantumRegister(N)
    qc = QuantumCircuit(qr)
    
    # Create H_init (Hamiltonian whose ground state is |0101>)
    ops_init = []
    for i in range(N):
        coeff = 1.0 if i % 2 == 1 else -1.0
        s = ['I'] * N
        s[N-1-i] = 'Z'
        ops_init.append(("".join(s), coeff))
    H_init = SparsePauliOp.from_list(ops_init)
    
    # Set initial state |0101>
    for i in range(N):
        if i % 2 == 1: 
            qc.x(i)
            
    # Time evolution
    dt = total_time / steps
    for s in range(steps):
        t = s / steps
        # Linear interpolation
        H_t = (1 - t) * H_init + t * H_target
        evo = PauliEvolutionGate(H_t, time=dt)
        # Apply 2nd order Trotter decomposition
        qc.append(SuzukiTrotter(order=2, reps=1).synthesize(evo), qr)
        
    return qc







# =====================================================================
# 2. [NEW] Iceberg-based Integrated GSP+GSM Extension Module
# =====================================================================

def build_logical_heisenberg_hamiltonian(N_data, J, h, total_qubits=6):
    """
    Constructs a logical Hamiltonian for the Heisenberg XYZ model on a lattice.
    Performs encoded operations by mapping physical qubits to logical qubits.
    """
    pauli_list = []
    coeffs = []
    Jx, Jy, Jz = J

    for i in range(N_data - 1):
        for op, coupling in zip(['X', 'Y', 'Z'], [Jx, Jy, Jz]):
            if coupling != 0.0:
                s = ['I'] * total_qubits
                s[total_qubits - 1 - i] = op
                s[total_qubits - 1 - (i + 1)] = op
                pauli_list.append("".join(s))
                coeffs.append(coupling)
                
    if h != 0.0:
        for i in range(N_data):
            s = ['I'] * total_qubits
            s[total_qubits - 1 - i] = 'Z'
            s[total_qubits - 1 - N_data] = 'Z' # position of parity[0]
            pauli_list.append("".join(s))
            coeffs.append(h)

    return SparsePauliOp(pauli_list, coeffs)

def build_iceberg_gsm_hamiltonian(total_qubits=6):
    """
    Constructs the Iceberg GSM (Ground State Metric) Hamiltonian for hardware-efficient simulation.
    Uses an optimized set of Pauli operators to reduce the depth of logical operations.
    """
    pauli_list = [
        "I" * total_qubits,
        "X" * total_qubits,
        "Z" * total_qubits,
        "Y" * total_qubits
    ]
    coeffs = [np.pi * 0.25, np.pi * 0.25, np.pi * 0.25, -np.pi * 0.25]
    return SparsePauliOp(pauli_list, coeffs)

def initialize_iceberg_logical_state(qc, q_data, q_parity):
    """
    Prepares the logical initial state suitable for the Iceberg lattice structure.
    """
    qc.h(q_parity[0])
    for i in range(4):
        qc.cx(q_parity[0], q_data[i])
    qc.cx(q_parity[0], q_parity[1])
    
    qc.x(q_data[1])
    qc.x(q_data[3])
    qc.barrier()

def create_integrated_mcmr_circuit(logical_hamiltonian, dt, degree, N_data=4, N_parity=2, trotter_reps=None):
    """
    Creates an integrated circuit that simultaneously performs GSP (filtering) and GSM (error detection)
    using a single ancilla qubit.
    """
    q_data = QuantumRegister(N_data, 'data')
    q_parity = QuantumRegister(N_parity, 'parity')
    q_anc = QuantumRegister(1, 'ancilla')
    
    c_anc = ClassicalRegister(degree, 'c_anc') 
    c_sys = ClassicalRegister(N_data + N_parity, 'c_sys') 
    
    qc = QuantumCircuit(q_anc, q_parity, q_data, c_anc, c_sys)
    
    initialize_iceberg_logical_state(qc, q_data, q_parity)
    
    all_logical_qubits = list(q_data) + list(q_parity)
    
    reps = _resolve_trotter_reps(dt, trotter_reps)
    trotter_gate = PauliEvolutionGate(logical_hamiltonian, time=dt, synthesis=SuzukiTrotter(order=2, reps=reps))
    c_trotter = trotter_gate.control(1)
    
    gsm_hamiltonian = build_iceberg_gsm_hamiltonian(N_data + N_parity)
    gsm_gate = PauliEvolutionGate(gsm_hamiltonian, time=-1.0, synthesis=SuzukiTrotter(order=2, reps=1))
    c_gsm = gsm_gate.control(1)
    
    for k in range(degree):
        qc.h(q_anc[0])
        # [Part A] GSP: Logical Trotter time evolution
        qc.append(c_trotter, [q_anc[0]] + all_logical_qubits)
        # [Part B] GSM: Codespace projection
        qc.append(c_gsm, [q_anc[0]] + all_logical_qubits)
        
        qc.h(q_anc[0])
        qc.measure(q_anc[0], c_anc[k])
        qc.reset(q_anc[0]) 
        qc.barrier()
        
    return qc, q_data, q_parity, c_sys


def create_hardware_efficient_mcmr_circuit(logical_hamiltonian, dt, degree, N_data=4, N_parity=2, trotter_reps=None):
    """
    [Manual Hardware-Efficient Design]
    Reuses 1 ancilla qubit 3 times to sequentially verify GSP, S_X, and S_Z.
    """
    q_data = QuantumRegister(N_data, 'data')
    q_parity = QuantumRegister(N_parity, 'parity')
    q_anc = QuantumRegister(1, 'ancilla')
    
    # Since 3 measurements occur per step, a total of degree * 3 classical bits are needed.
    c_meas = ClassicalRegister(degree * 3, 'c_meas') 
    c_sys = ClassicalRegister(N_data + N_parity, 'c_sys') 
    
    qc = QuantumCircuit(q_anc, q_parity, q_data, c_meas, c_sys)
    
    # Prepare logical initial state |0101> (same as before)
    initialize_iceberg_logical_state(qc, q_data, q_parity)
    
    all_logical_qubits = list(q_data) + list(q_parity)
    
    reps = _resolve_trotter_reps(dt, trotter_reps)
    trotter_gate = PauliEvolutionGate(logical_hamiltonian, time=dt, synthesis=SuzukiTrotter(order=2, reps=reps))
    c_trotter = trotter_gate.control(1)
    
    meas_idx = 0
    for k in range(degree):
        # ------------------------------------------------
        # [1] GSP Part: Ground State Projection (Filter)
        # ------------------------------------------------
        qc.h(q_anc[0])
        qc.append(c_trotter, [q_anc[0]] + all_logical_qubits)
        qc.h(q_anc[0])
        qc.measure(q_anc[0], c_meas[meas_idx])
        qc.reset(q_anc[0])
        meas_idx += 1
        
        # ------------------------------------------------
        # [2] Iceberg Part: X Stabilizer (S_X) Error Detection
        # ------------------------------------------------
        qc.h(q_anc[0])
        for q in all_logical_qubits:
            qc.cx(q_anc[0], q) # O(1) depth implementation with only 6 CNOTs
        qc.h(q_anc[0])
        qc.measure(q_anc[0], c_meas[meas_idx])
        qc.reset(q_anc[0])
        meas_idx += 1
        
        # ------------------------------------------------
        # [3] Iceberg Part: Z Stabilizer (S_Z) Error Detection
        # ------------------------------------------------
        qc.h(q_anc[0])
        for q in all_logical_qubits:
            qc.cz(q_anc[0], q) # O(1) depth implementation with only 6 CZs
        qc.h(q_anc[0])
        qc.measure(q_anc[0], c_meas[meas_idx])
        qc.reset(q_anc[0])
        meas_idx += 1
        
        qc.barrier()
        
    return qc, q_data, q_parity, c_sys



def add_logical_basis_measurement(base_qc, q_data, q_parity, c_sys, basis):
    """Measures both data and parity qubits after logical basis transformation."""
    qc = base_qc.copy()
    if basis == 'X':
        qc.h(q_data)
    elif basis == 'Y':
        qc.sdg(q_data)
        qc.h(q_data)
        
    qc.measure(list(q_data) + list(q_parity), c_sys)
    return qc

def calculate_integrated_energy_v2(job_result, bases, N_data, J, h):
    """
    Receives SamplerV2 results and performs post-selection on a shot-by-shot basis.
    Pass condition: All ancilla qubits in the trajectory are '0'.
    """
    Jx, Jy, Jz = J
    expected_energy = 0.0
    total_variance = 0.0  
    passed_shots_info = {}

    for i, basis in enumerate(bases):
        pub_result = job_result[i]
        sys_bitstrings = pub_result.data.c_sys.get_bitstrings()
        anc_bitstrings = pub_result.data.c_anc.get_bitstrings()
        
        shot_energies = [] 
        valid_shots = 0
        
        for sys_bits, anc_bits in zip(sys_bitstrings, anc_bitstrings):
            if '0' not in anc_bits: # Perfect trajectory evolved to the ground state without errors
                valid_shots += 1
                physical_bits = sys_bits[::-1] 
                
                shot_energy = 0.0
                for idx in range(N_data - 1):
                    val = 1 if physical_bits[idx] == physical_bits[idx+1] else -1
                    if basis == 'X': shot_energy += Jx * val
                    elif basis == 'Y': shot_energy += Jy * val
                    elif basis == 'Z': shot_energy += Jz * val
                
                if basis == 'Z' and h != 0.0:
                    for idx in range(N_data):
                        z_i = 1 if physical_bits[idx] == '0' else -1
                        z_p0 = 1 if physical_bits[N_data] == '0' else -1 # Position of parity[0]
                        shot_energy += h * (z_i * z_p0)
                        
                shot_energies.append(shot_energy)
                
        if valid_shots > 1:
            expected_energy += np.mean(shot_energies)
            total_variance += np.var(shot_energies, ddof=1) / valid_shots 
        elif valid_shots == 1:
            expected_energy += shot_energies[0]
            
        passed_shots_info[basis] = valid_shots

    total_passed = passed_shots_info.get('Z', 0)
    standard_error = np.sqrt(total_variance) if total_passed > 0 else 0.0
    
    return expected_energy, standard_error, total_passed

def calculate_hardware_efficient_energy_v2(job_result, bases, N_data, J, h):
    """
    Updated post-selection logic. 
    Now, success is achieved only when all ancilla results (GSP, X detection, Z detection) are perfectly '0'.
    """
    Jx, Jy, Jz = J
    expected_energy = 0.0
    total_variance = 0.0  
    passed_shots_info = {}

    for i, basis in enumerate(bases):
        pub_result = job_result[i]
        sys_bitstrings = pub_result.data.c_sys.get_bitstrings()
        anc_bitstrings = pub_result.data.c_meas.get_bitstrings() # Changed to c_meas
        
        shot_energies = [] 
        valid_shots = 0
        
        for sys_bits, anc_bits in zip(sys_bitstrings, anc_bitstrings):
            # [Core Modification] Discard if even a single '1' (error occurred or filtering failed) is present
            if '1' not in anc_bits: 
                valid_shots += 1
                physical_bits = sys_bits[::-1] 
                
                shot_energy = 0.0
                for idx in range(N_data - 1):
                    val = 1 if physical_bits[idx] == physical_bits[idx+1] else -1
                    if basis == 'X': shot_energy += Jx * val
                    elif basis == 'Y': shot_energy += Jy * val
                    elif basis == 'Z': shot_energy += Jz * val
                
                if basis == 'Z' and h != 0.0:
                    for idx in range(N_data):
                        z_i = 1 if physical_bits[idx] == '0' else -1
                        z_p0 = 1 if physical_bits[N_data] == '0' else -1
                        shot_energy += h * (z_i * z_p0)
                        
                shot_energies.append(shot_energy)
                
        if valid_shots > 1:
            expected_energy += np.mean(shot_energies)
            total_variance += np.var(shot_energies, ddof=1) / valid_shots 
        elif valid_shots == 1:
            expected_energy += shot_energies[0]
            
        passed_shots_info[basis] = valid_shots

    total_passed = passed_shots_info.get('Z', 0)
    standard_error = np.sqrt(total_variance) if total_passed > 0 else 0.0
    
    return expected_energy, standard_error, total_passed