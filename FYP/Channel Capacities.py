import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


bases = ["X", "Z"]
# Z basis Matrices
Z0 = 1 / np.sqrt(2) * np.array([[1], [1]])
Z1 = 1 / np.sqrt(2) * np.array([[1], [-1]])
# Z gate
Z = np.concatenate((Z0, Z1), axis=1)

# X basis Matrices
X0 = np.array([[1], [0]])
X1 = np.array([[0], [1]])
# X gate
X = np.concatenate((X0, X1), axis=1)

BitFile1 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults1.csv'
BitFile2 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults2.csv'
BitFile3 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults3.csv'
BitFile4 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults4.csv'

BitResults1 = pd.read_csv(BitFile1, dtype=str)
BitResults2 = pd.read_csv(BitFile2, dtype=str)
BitResults3 = pd.read_csv(BitFile3, dtype=str)
BitResults4 = pd.read_csv(BitFile4, dtype=str)

data_df = pd.concat([BitResults1, BitResults2, BitResults3, BitResults4])

qrng = np.array(data_df['QRNG'].astype(int))
random_module = np.array(data_df['Random Module'].astype(int))
ibmq = np.array(data_df['IBMQ'].astype(int))

class QuantumChannel:
    def __init__(self, distance, attenuation_coefficient, phase_noise_std, polarization_fidelity):
        self.distance = distance
        self.attenuation_coefficient = attenuation_coefficient
        self.phase_noise_std = phase_noise_std
        self.polarization_fidelity = polarization_fidelity

    def calculate_attenuation(self):
        return np.exp(-self.attenuation_coefficient * self.distance)

    def apply_noise(self, qubit):
        attenuation = self.calculate_attenuation()
        attenuated_qubit = np.sqrt(attenuation) * qubit

        phase_noise = np.random.normal(0, self.phase_noise_std, size=qubit.shape)
        noisy_qubit = attenuated_qubit * np.exp(1j * phase_noise)

        polarization_noise = np.random.choice([0, 1], size=qubit.shape, p=[self.polarization_fidelity, 1 - self.polarization_fidelity])
        noisy_qubit = polarization_noise * noisy_qubit

        return noisy_qubit
class PublicChannel:
    def __init__(self):
        pass

    def transmit(self, message):
        return message
class Qubit:
    def __init__(self, bit, base):
        self.bit = bit
        self.base = base
        self.qubit = self.encode_qubit()

    def encode_qubit(self):
        if self.bit == 0:
            qubit = np.array([[1], [0]])  # |0>
        elif self.bit == 1:
            qubit = np.array([[0], [1]])  # |1>
        else:
            raise ValueError("Invalid bit value")

        if self.base == "X":
            qubit = np.dot(X, qubit)  # Apply X gate if base is X
        elif self.base == "Z":
            qubit = np.dot(Z, qubit)  # Apply Z gate if base is Z
        else:
            raise ValueError("Invalid base value")

        return qubit

    def decode_qubit(self, qubit):
        try:
            if self.base == "X":
                qubit = X @ qubit
            elif self.base == "Z":
                qubit = Z @ qubit

            measurement = np.abs(qubit) ** 2 > 0.5

            return np.any(measurement)

        except ValueError:
            pass
class Sender:
    def __init__(self, data):
        self.data = data

    def generate_message(self, n):
        alice_bits = self.data[:n]
        alice_bases = [random.choice(bases) for _ in range(n)]
        alice_qubits = [Qubit(bit, base) for bit, base in zip(alice_bits, alice_bases)]

        return alice_qubits, alice_bases, alice_bits
class Receiver:
    def __init__(self):
        pass

    def measure_qubits(self, qubits, bases):
        received_qubits = []
        measurements = []
        bob_bases = {}
        bob_bits = []
        for i, (qubit, base) in enumerate(zip(qubits, bases)):
            # measure the qubit in the eve base
            measurement = Qubit(0, base).decode_qubit(qubit)
            measurements.append(measurement)

            bob_bases[i] = base
            # encode qubits in eve base
            if bob_bases[i] == base:
                received_qubits.append(Qubit(measurement, base).qubit)
            else:
                received_qubits.append(Qubit(1 - measurement, base).qubit)

        for i, measurement in enumerate(measurements):
            if measurement == 1:
                bob_bits.append(alice_bits[i])
            else:
                bob_bits.append(random.randint(0, 1))

        return np.array(received_qubits), bob_bases, measurements, bob_bits

class Eavesdropper:
    def __init__(self,eavesdropping_probabilty):
        self.eavesdropping_probabilty=eavesdropping_probabilty

    def intercept_and_forward(self, qubits, bases, alice_bits):
        intercepted_qubits = []
        measurements = []
        eve_bases = {}
        eve_bits = []
        for i, (qubit, base) in enumerate(zip(qubits, bases)):
            if random.random() >= self.eavesdropping_probabilty:
                # measure the qubit in the eve base
                measurement = Qubit(0, base).decode_qubit(qubit)
                measurements.append(measurement)

                eve_bases[i] = base
                # encode qubits in eve base
                if eve_bases[i] == base:
                    intercepted_qubits.append(Qubit(measurement, base).qubit)
                else:
                    intercepted_qubits.append(Qubit(1 - measurement, base).qubit)
            else:
                intercepted_qubits.append(qubit)

        for i, measurement in enumerate(measurements):
            if measurement == 1:
                eve_bits.append(alice_bits[i])
            else:
                eve_bits.append(random.randint(0, 1))

        return np.array(intercepted_qubits), eve_bases, measurements, eve_bits
class ErrorReconciliation:
    def __init__(self,error_probability):
        self.error_probability=error_probability

    def parity_check(self, alice_sifted_key, bob_sifted_key):
        parity_bits = []
        for alice_bit, bob_bit in zip(alice_sifted_key, bob_sifted_key):
            parity_bits.append(alice_bit ^ bob_bit)  # Compute parity bit

        return parity_bits

    def correct_errors(self, sifted_key, parity_bits):
        corrected_key = []
        for bit, parity in zip(sifted_key, parity_bits):
            if parity == 0:  # No error detected
                corrected_key.append(bit)
            else:  # Error detected, flip the bit
                if random.random() >= self.error_probability:
                    corrected_key.append(1- bit)
                else:
                    corrected_key.append(bit)

        return corrected_key
    
def calculate_entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy += -p * np.log2(p)
    return entropy

def calculate_vonneumanentropy(density_matrices):
    entropy = 0
    for rho in density_matrices:
        if rho.shape[0] != rho.shape[1]:
            raise ValueError("Input density matrices must be square.")
        
        eigenvalues = np.linalg.eigvalsh(rho)
        non_zero_eigenvalues = eigenvalues[eigenvalues > 0]
        if len(non_zero_eigenvalues) > 0:
            entropy -= np.sum(non_zero_eigenvalues * np.log2(non_zero_eigenvalues))
    return entropy

def calculate_conditional_entropy(alice_bits, bob_measurements):
    unique_bits = np.unique(alice_bits)
    unique_measurements = np.unique(bob_measurements)
    conditional_entropy = 0.0

    for bit in unique_bits:
        bit_indices = np.where(alice_bits == bit)[0]
        bit_measurements = bob_measurements[bit_indices]
        measurement_counts = np.array([np.sum(bit_measurements == m) for m in unique_measurements])
        measurement_probs = measurement_counts / len(bit_measurements)
        conditional_entropy += np.sum(-measurement_probs * np.log2(measurement_probs))

    return conditional_entropy


"""# Define the parameter values to loop over"""
n = 500
distances = np.arange(0,1000,10)
attenuation_coefficient = 0.5
phase_noise_std = 0.2
fidelity = 0.1
eavesdropping_rate = 0.5

"""Arrays to store data of keys"""
data_loss_rates = []  # List to store the data loss rates
error_rates = [] # List to store the error rates
channel_capacities = [] # List to store the channel capacities
key_rates = [] # List to store the key rates
entropies=[] # List to store the entropies
mutual_informations=[] # List to store the mutual informations
conditional_entropies=[] # List to store the conditional entropies


"""# Run the protocol for different distances"""
"""QKD Protocol"""
for distance in distances:
    alice_sifted_key = []
    bob_sifted_key = []
    final_sifted_key = []
    new_corrected_key = []
    matching_bits = []
    matching_bases = []
    matching_indices = []
    
    """# Generate random bit sequence for the message"""
    sender = Sender(qrng)
    alice_qubits, alice_bases, alice_bits = sender.generate_message(n)

    """# Transmit Qubits over Quantum Channel"""
    quantum_channel = QuantumChannel(distance, attenuation_coefficient, phase_noise_std, fidelity)
    noisy_qubits = [quantum_channel.apply_noise(q.qubit) for q in alice_qubits]

    alice_qubits = np.array([qubit.qubit for qubit in alice_qubits])

    """# Initialize Eve"""
    eve = Eavesdropper(eavesdropping_rate)
    eve_bases = [random.choice(bases) for _ in range(n)]
    intercepted_qubits, _, intercepted_measurements, intercepted_bits = eve.intercept_and_forward(
        noisy_qubits, eve_bases, alice_bits
    )

    """# Transmit the bases over the public channel"""
    public_channel = PublicChannel()
    bob_bases = [random.choice(bases) for _ in range(n)]

    """# Measure the qubits based on the received bases"""
    receiver = Receiver()
    bob_qubits, bob_bases, bob_measurements, bob_bits = receiver.measure_qubits(intercepted_qubits, bob_bases)


    """##QKD Transmission Complete"""

    """Sifting"""
    for i in range(len(alice_bits)):
        if alice_bits[i] == bob_bits[i]:
            matching_bits.append(alice_bits[i])
        if alice_bases[i] == bob_bases[i]:
            matching_bases.append(alice_bases[i])
            matching_indices.append(i)
            alice_sifted_key.append(alice_bits[i])
            bob_sifted_key.append(bob_bits[i])
            if bob_bits[i] == alice_bits[i]:
                final_sifted_key.append(alice_bits[i])
                
    """Error Reconciliation"""
    error_reconciliation = ErrorReconciliation(0.5)
    parity_bits = error_reconciliation.parity_check(alice_sifted_key, bob_sifted_key)
    corrected_key = error_reconciliation.correct_errors(bob_sifted_key, parity_bits)
    for i in range(len(alice_sifted_key)):
        if corrected_key[i] == alice_sifted_key[i]:
            new_corrected_key.append(corrected_key[i])
            
    """Data Loss and Error Rate"""
    data_loss = (n - len(new_corrected_key)) / n
    error_rate = (len(alice_sifted_key) - len(new_corrected_key)) / (len(alice_sifted_key))

    alice_bits = np.array(alice_bits)
    bob_measurements = np.array(bob_bits)

    conditional_entropy = calculate_conditional_entropy(alice_bits, bob_measurements)
    # Calculate Alice's key probabilities
    p0 = len(alice_bits[alice_bits == 0]) / len(alice_bits)
    p1 = len(alice_bits[alice_bits == 1]) / len(alice_bits)

    # Calculate entropy of Alice's key
    entropy = calculate_entropy([p0, p1])
    # print(bob_qubits)
    # vonnentropy=calculate_vonneumanentropy(bob_qubits)

    # Calculate mutual information
    mutual_information = entropy - conditional_entropy

    # Calculate channel capacity
    channel_capacity = entropy - mutual_information

    channel_capacities.append(channel_capacity)
    data_loss_rates.append(data_loss)
    error_rates.append(error_rate)
    key_rates.append(channel_capacity * (1 - data_loss))
    entropies.append(entropy)
    mutual_informations.append(mutual_information)
    conditional_entropies.append(conditional_entropy)
    
plt.figure(figsize=(10, 6))
plt.plot(distances, data_loss_rates, label="Data Loss Rate")
plt.plot(distances, error_rates, label="Error Rate")
plt.plot(distances, channel_capacities, label="Channel Capacity")
plt.plot(distances, entropies, label="Entropy")
plt.plot(distances, key_rates, label="Key Rate")
plt.xlabel("Distance (km)")
plt.ylabel("Rates")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()



    
