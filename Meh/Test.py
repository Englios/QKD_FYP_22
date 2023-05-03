import random
import numpy as np

# Define the size of the message to be sent
n = 10

# Define the bases used for encoding and measuring the qubits
bases = ["X", "Z"]

# Define the quantum gates used for encoding the qubits
X_GATE = np.array([[0, 1], [1, 0]])
Z_GATE = np.array([[1, 0], [0, -1j]])

# Define the quantum channel, which applies a noise operator to the qubits
def quantum_channel(qubit, noise_level):
    """Simulates the quantum channel that applies noise to the qubits"""
    error = np.random.rand() < noise_level
    if error:
        pauli_error = random.choice([X_GATE, Z_GATE])
        noisy_qubit = pauli_error @ qubit
    else:
        noisy_qubit = qubit
    return noisy_qubit

# Define the public channel, which transmits the bases used for encoding and measuring the qubits
def public_channel(message):
    """Simulates the public channel that transmits the message"""
    return message

# Define the encoding and decoding functions for the qubits
def encode_qubit(bit, base):
    """Encodes a qubit based on the bit and base"""
    qubit = np.array([1, 0])
    if bit == 1:
        qubit = X_GATE @ qubit
    if base == "Z":
        qubit = Z_GATE @ qubit
    return qubit

def decode_qubit(qubit, base):
    """Decodes a qubit based on the base"""
    if base == "X":
        qubit = X_GATE @ qubit
    if base == "Z":
        qubit = Z_GATE @ qubit
    measurement = np.abs(qubit[0])**2 > 0.5
    return measurement

# Generate random bit sequence for the message
alice_bits = [random.randint(0, 1) for i in range(n)]

# Encode the message into qubits
alice_bases = [random.choice(bases) for i in range(n)]
alice_qubits = [encode_qubit(bit, base) for bit, base in zip(alice_bits, alice_bases)]

# Transmit the qubits over the quantum channel
noise_level = 0.5
bob_qubits = [quantum_channel(qubit, noise_level) for qubit in alice_qubits]

# Transmit the bases over the public channel
bob_bases = [random.choice(bases) for i in range(n)]

# Measure the qubits based on the received bases
bob_measurements = [decode_qubit(qubit, base) for qubit, base in zip(bob_qubits, bob_bases)]

# Determine the matching bits
matching_bits = [i for i in range(n) if alice_bases[i] == bob_bases[i]]

# Use the matching bits as the key
key = [alice_bits[i] for i in matching_bits if alice_bases[i] == bob_bases[i]]

print("Alice's bits:", alice_bits)
print("Alice's bases:", alice_bases)
print("Bob's qubits:", bob_qubits)
print("Bob's bases:", bob_bases)
print("Bob's measurements:", bob_measurements)
print("Shared key:", key)
