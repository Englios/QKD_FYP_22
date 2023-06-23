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
            qubit = np.array([1, 0])  # |0>
        elif self.bit == 1:
            qubit = np.array([0, 1])  # |1>
        else:
            raise ValueError("Invalid bit value")

        if self.base == "X":
            qubit = X @ qubit  # Apply X gate if base is X
        elif self.base == "Z":
            qubit = Z @ qubit  # Apply Z gate if base is Z
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

        return received_qubits, bob_bases, measurements, bob_bits
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

        return intercepted_qubits, eve_bases, measurements, eve_bits
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

"""# Define the parameter values to loop over"""
n = 500
distance = 100
attenuation_coefficient = 0.5
phase_noise_std = 0.5
fidelity = 0.1
eavesdropping_rates = np.linspace(0, 1, 250)

"""Arrays to store data of keys"""

data_loss_rates = []  # List to store the data loss rates
error_rates = [] # List to store the error rates

matching_bases_ratio=[]
mismatch_bases_ratio=[]

matching_bits_ratio=[]
mismatching_bits_ratio=[]
matching_bits_before_sifting_ratio = []
matching_bits_after_sifting_ratio = []
mismacthing_bits_before_sifting_ratio = []
mismatching_bits_after_sifting_ratio = []

for eavesdopping_rate in eavesdropping_rates:
    matching_bases=[]
    matching_bits=[]
    matching_indices=[]
    matching_bits_before_sifting = 0
    matching_bits_after_sifting = 0
    
    """QKD Protocol"""
    alice_sifted_key = []
    bob_sifted_key = []
    final_sifted_key = []
    new_corrected_key = []
    """# Generate random bit sequence for the message"""
    sender = Sender(qrng)
    alice_qubits, alice_bases, alice_bits = sender.generate_message(n)

    """# Transmit Qubits over Quantum Channel"""
    quantum_channel = QuantumChannel(distance, attenuation_coefficient, phase_noise_std, fidelity)
    noisy_qubits = [quantum_channel.apply_noise(q.qubit) for q in alice_qubits]

    alice_qubits = np.array([qubit.qubit for qubit in alice_qubits])

    """# Initialize Eve"""
    eve = Eavesdropper(eavesdopping_rate)
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
    """# Error Reconciliation"""
    for i in range(len(alice_bits)):
        if alice_bits[i] == bob_bits[i]:
            matching_bits_before_sifting += 1

    matching_bits_before_sifting = matching_bits_before_sifting/ len(alice_bits)
    
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

    matching_bits_after_sifting = len(new_corrected_key) / len(alice_sifted_key)
    
                
    """Data Loss and Error Rate"""

    data_loss = (n - len(bob_sifted_key)) / n
    error_rate = (len(bob_sifted_key) - len(final_sifted_key)) / (len(bob_sifted_key))

    # # Info Printing
    # print('Eavesdropping Rate:', eavesdopping_rate)
    # print('Matching Bases Ratio:', len(matching_bases) / len(alice_bases))
    # print('Mismatch Bases Ratio:', 1 - (len(matching_bases) / len(alice_bases)))
    # print('Sifted Key Length:', len(matching_indices))
    # print('Data Loss Rate:', data_loss)
    # print('Error Rate:', error_rate)
    # print('---Before Sifting---')
    # print('Matching Bits:', matching_bits_before_sifting)
    # print('Mismatch Bits:', 1-matching_bits_before_sifting)
    # print('---After Sifting---')
    # print('Matching Bits:', matching_bits_after_sifting)
    # print('Mismatch Bits:', 1-matching_bits_after_sifting)
    # print('----------------------')

    # Store the results in the corresponding lists
    matching_bases_ratio.append(len(matching_bases) / len(alice_bases))
    mismatch_bases_ratio.append(1 - (len(matching_bases) / len(alice_bases)))
    data_loss_rates.append(data_loss)
    error_rates.append(error_rate)
    
    matching_bits_before_sifting_ratio.append(matching_bits_before_sifting)
    matching_bits_after_sifting_ratio.append(matching_bits_after_sifting)
    mismacthing_bits_before_sifting_ratio.append(1-matching_bits_before_sifting)
    mismatching_bits_after_sifting_ratio.append(1-matching_bits_after_sifting) 
    
"""# Plot the results"""
plt.figure(figsize=(10, 6))
plt.plot(eavesdropping_rates, matching_bits_after_sifting_ratio, color='red', linestyle='--',alpha=0.8, label="Matching Bits After Sifting")
plt.plot(eavesdropping_rates, matching_bits_before_sifting_ratio, color='red', linestyle='-.',alpha=0.6,label="Matching Bits Before Sifting")
plt.plot(eavesdropping_rates, matching_bases_ratio, color='red', label="Matching Bases Ratio")
plt.plot(eavesdropping_rates, mismatch_bases_ratio, color='blue', label="Mismatch Bases Ratio")
plt.plot(eavesdropping_rates, mismacthing_bits_before_sifting_ratio, color='blue', linestyle='-.',alpha=0.6, label="Mismatching Bits Before Sifting")
plt.plot(eavesdropping_rates, mismatching_bits_after_sifting_ratio, color='blue', linestyle='--',alpha=0.8, label="Mismatching Bits After Sifting")

plt.xlabel("Eavesdropping Rate",size=20)
plt.ylabel("Match and Mismatch Ratio",size=20)
plt.yticks(np.arange(0, 1, 0.05),size=14)
plt.xticks(np.arange(0, 1, 0.05),size=14,rotation=275)
plt.title("Matching/Mismatching bases and bits ratio vs Eavesdropping Rate")
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(eavesdropping_rates, error_rates, color='limegreen', label="Error Rate")

plt.xlabel("Eavesdropping Rate")
plt.ylabel("Error")
plt.title("Error as a function of Eve")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()   

# Calculate mean and variance of data loss rates
mean_data_loss = np.mean(data_loss_rates)
var_data_loss = np.var(data_loss_rates)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(eavesdropping_rates, data_loss_rates, color='salmon', linestyle='--', label="Data Loss Rate", alpha=0.8)
plt.scatter(eavesdropping_rates, data_loss_rates, color='red', marker='.', label="Data Loss Rate")
plt.axhline(mean_data_loss, color='black', linestyle='-', label="Mean Data Loss Rate")

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(eavesdropping_rates, data_loss_rates)
regression_line = slope * eavesdropping_rates + intercept
plt.plot(eavesdropping_rates, regression_line, color='blue', linestyle='-', label="Linear Regression", linewidth=3)

plt.xlabel("Eavesdropping Rate")
plt.ylabel("Data Loss Rate")
plt.title("Data Loss Rate vs Eavesdropping Rate")
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
    
    
