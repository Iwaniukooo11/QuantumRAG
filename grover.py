from math import ceil, log2, pi, floor, asin, sqrt
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
import qiskit
import qiskit_ibm_runtime  
from qiskit_aer import Aer


def grover_top_k(results, k=3, initial_threshold=0.75, shots=10000):
    """
    Selects top-k contexts from the `results` list using Grover,
    based on whether similarity_score > threshold, dynamically adjusting the threshold.
    Returns the top-k results (and additional info).
    """
    n_items = len(results)
    if n_items == 0 or k == 0:
        return []

    n_qubits = ceil(log2(n_items))
    threshold = initial_threshold

    backend = Aer.get_backend('aer_simulator')

    def grover_oracle(marked_indices):
        binary_states = [format(i, f"0{n_qubits}b") for i in marked_indices]
        qc = QuantumCircuit(n_qubits)
        for target in binary_states:
            rev_target = target[::-1]
            zero_inds = [ind for ind in range(n_qubits) if rev_target[ind] == "0"]
            qc.x(zero_inds)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits)))
            qc.x(zero_inds)
        return qc

    def diffusion_operator(n):
        qc = QuantumCircuit(n)
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.append(MCXGate(n - 1), qargs=list(range(n)))
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))
        return qc

    # Dynamically adjust the threshold
    step = 0.1
    min_step = 0.001
    max_attempts = 100
    attempts = 0
    marked_indices = []
    while attempts < max_attempts:
        marked_indices = [i for i, res in enumerate(results) if res["similarity_score"] > threshold]
        if len(marked_indices) > k:
            threshold += step
            step = max(step / 2, min_step)
        elif len(marked_indices) < k:
            threshold -= step
            step = max(step / 2, min_step)
        else:
            break
        attempts += 1

    if len(marked_indices) == 0:
        return []

    oracle = grover_oracle(marked_indices)
    diffuser = diffusion_operator(n_qubits)

    optimal_num_iterations = floor(
        pi / (4 * asin(sqrt(len(marked_indices) / 2**n_qubits)))
    )
    qc = QuantumCircuit(n_qubits)
    # Create even superposition of all basis states
    qc.h(range(n_qubits))
    for _ in range(optimal_num_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    # Measure all qubits
    qc.measure_all()

    result = backend.run(qc, shots=shots).result()
    dist = result.get_counts()


    # Sort the measured states by their counts (probabilities)
    sorted_dist = sorted(dist.items(), key=lambda item: item[1], reverse=True)

    # Return the top-k results from the original 'results' list
    top_k_results = []
    for state, _ in sorted_dist:
        idx = int(state, 2)
        if 0 <= idx < len(results) and results[idx]["similarity_score"] > threshold:
            top_k_results.append(results[idx])
        if len(top_k_results) == k:
            break

    return {
        "contexts": top_k_results,
        "threshold": threshold,
        "qiskit_version": qiskit.__version__,
        "runtime_version": qiskit_ibm_runtime.__version__
    }
