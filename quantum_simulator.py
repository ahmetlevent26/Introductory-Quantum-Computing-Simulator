import numpy as np

# Define qubit states |0> and |1> as complex vectors

zero = np.array([1.0, 0.0], dtype=complex)
one = np.array([0.0, 1.0], dtype=complex)

# Degine qubit gates as 2x2 complex matrices

# Identity Gate: does nothing to the qubit

I = np.eye(2, dtype=complex)

# Pauli-X Gate: flips |0> to |1> and |1> to |0>

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

#Pauli-Z Gate: applies a phase flip to |1>

Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

#Hadamard Gate: createse a superpostion of |0> and |1>

H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                 [1, -1]], dtype=complex)

#Define the Kronecker Product, which is combines multiple qubit states or gates

def kron_n(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

#Define n-qubit basis state, which constructs a multi-qubit state from a bitstring

def n_qubit_basis_state(bits):
    state = zero if bits[0] == "0" else one
    for b in bits[1:]:
        state = np.kron(state, zero if b == "0" else one)
    return state

#Define single-qubit gate applied to n-qubit system, which applies a gate to a specific qubit in a multi-qubit system

def single_qubit_gate_on_n_qubits(gate, n, target):
    ops = []
    for i in range(n):
        ops.append(gate if i == target else I)
    return kron_n(ops)

#Define a function to apply a gate to a quantum state

def apply_gate(state, gate_op):
    return gate_op @ state

#Define a function to measure probabilities of each basis state in a quantum state

def measure_probabilities(state):
    dim = state.shape[0]
    n = int(np.log2(dim))
    probs = np.abs(state) ** 2

    results = []
    for idx, p in enumerate(probs):
        bitstring = format(idx, f"0{n}b")
        results.append((bitstring, float(p)))
    return sorted(results, key=lambda x: x[1], reverse=True)

# Demo 1: Single-qubit superposition
# Start with |0>, apply H, and measure probabilities

def demo_single_qubit_superposition():
    print("=== Single-qubit superposition demo ===")
    state = zero
    print("Initial state: |0>")

    state = apply_gate(state, H)

    probs = measure_probabilities(state)
    print("Probabilities after H on |0>:")
    for bitstring, p in probs:
        print(f"  |{bitstring}>: {p:.3f}")
    print()

#Demo 2: Two-qubit Bell state
# Start with |00>, apply H to qubit 0 (|00> + |10>)/sqrt(2), then CNOT to entangle

def demo_two_qubit_bell_state():
    print("=== Two-qubit Bell state demo ===")

    state = n_qubit_basis_state("00")
    print("Initial state: |00>")

    H0 = single_qubit_gate_on_n_qubits(H, n=2, target=0)
    state = apply_gate(state, H0)

    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

    state = apply_gate(state, CNOT)

    probs = measure_probabilities(state)
    print("Probabilities for Bell state:")
    for bitstring, p in probs:
        print(f"  |{bitstring}>: {p:.3f}")
    print()


def main():
    demo_single_qubit_superposition()
    demo_two_qubit_bell_state()


if __name__ == "__main__":
    main()
