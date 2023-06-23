import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from PIL import Image

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

# AES encryption class
class AES:
    def __init__(self, key):
        self.key = self.derive_key(key)

    def derive_key(self, key):
        salt = b'salt'  # Salt value for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256-bit key size
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        derived_key = kdf.derive(key.encode('utf-8'))
        return derived_key

    def encrypt(self, message):
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(message) + padder.finalize()

        cipher = Cipher(algorithms.AES(self.key), modes.ECB(), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

        return encrypted_data

    def decrypt(self, encrypted_data):
        cipher = Cipher(algorithms.AES(self.key), modes.ECB(), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        decrypted_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()

        return decrypted_data

"""# Define the parameter values to loop over"""
n_runs = 10
n = 15
distance = 100

attenuation_coefficient = 0.5
phase_noise_std = 0.5
fidelity = 0.5

"""# Initialize arrays to store the data"""
error_rates = []
data_loss_rates = []
final_key_lengths = []

"""Arrays to store data of keys"""
matching_bases = []
matching_bases2 = []
matching_indices = []
matching_indices2 = []
bob_sifted_key = []
bob_sifted_key2 =[]
alice_sifted_key = []
final_sifted_key = []
final_sifted_key2 = []
uncorrected_sifted_key = []

"""# Run the simulation"""
"""# Generate random bit sequence for the message"""
sender = Sender(qrng)
alice_qubits, alice_bases, alice_bits = sender.generate_message(n)

"""# Transmit Qubits over Quantum Channel"""
quantum_channel = QuantumChannel(distance, attenuation_coefficient, phase_noise_std, fidelity)
noisy_qubits = [quantum_channel.apply_noise(q.qubit) for q in alice_qubits]

alice_qubits = np.array([qubit.qubit for qubit in alice_qubits])

"""# Initialize Eve"""
eve = Eavesdropper(0.5)
eve_bases = [random.choice(bases) for _ in range(n)]
intercepted_qubits, _, intercepted_measurements, intercepted_bits = eve.intercept_and_forward(
    noisy_qubits, eve_bases, alice_bits
)

"""# Transmit the bases over the public channel"""
public_channel = PublicChannel()
bob_bases = [random.choice(bases) for _ in range(n)]
bob_bases2 = bob_bases.copy()

"""# Measure the qubits based on the received bases"""
receiver = Receiver()
bob_qubits, bob_bases, bob_measurements, bob_bits = receiver.measure_qubits(alice_qubits, bob_bases)
bob_qubits2, bob_bases2, bob_measurements2, bob_bits2 = receiver.measure_qubits(intercepted_qubits, bob_bases2)

for i in range(len(alice_bits)):
    if alice_bases[i] == bob_bases[i]:
        matching_bases.append(alice_bases[i])
        matching_indices.append(i)
        alice_sifted_key.append(alice_bits[i])
        bob_sifted_key.append(bob_bits[i])
        if bob_bits[i] == alice_bits[i]:
            uncorrected_sifted_key.append(alice_bits[i])

for i in range(len(alice_bits)):
    if alice_bases[i] == bob_bases2[i]:
        matching_bases2.append(alice_bases[i])
        matching_indices2.append(i)
        bob_sifted_key2.append(bob_bits2[i])
        if bob_bits2[i] == alice_bits[i]:
            uncorrected_sifted_key.append(alice_bits[i])
            
"""# Error Reconciliation"""
error_reconciliation = ErrorReconciliation(0.5)
parity_bits = error_reconciliation.parity_check(alice_sifted_key, bob_sifted_key)
corrected_key = error_reconciliation.correct_errors(bob_sifted_key, parity_bits)
for i in range(len(alice_sifted_key)):
    if corrected_key[i] == alice_sifted_key[i]:
        final_sifted_key.append(corrected_key[i])

error_reconciliation = ErrorReconciliation(0.5)
parity_bits = error_reconciliation.parity_check(alice_sifted_key, bob_sifted_key2)
corrected_key2 = error_reconciliation.correct_errors(bob_sifted_key2, parity_bits)
for i in range(len(alice_sifted_key)):
    if corrected_key2[i] == alice_sifted_key[i]:
        final_sifted_key2.append(corrected_key2[i])
        

"""# Generate random key for encryption"""
"""Key Printing """
print('Alice Bits \n', alice_bits)
print('Alice Bases \n', alice_bases)
print('Bob Bits \n', bob_bits)
print('Bob Bases \n', bob_bases)
print('Matching Bases \n', matching_bases)
print('Alice Sifted Key \n', alice_sifted_key)
print('Bob Sifted Key \n', bob_sifted_key)
print('Corrected Key \n', corrected_key)
print('Final Sifted Key \n', final_sifted_key)

"""With Eve"""
print("\nWith Eve\n")
print('Alice Bits \n', alice_bits)
print('Alice Bases \n', alice_bases)
print('Eve Bases \n', eve_bases)
print('Eve Bits \n', intercepted_bits)
print('Bob Bits \n', bob_bits2)
print('Bob Bases \n', bob_bases2)
print('Matching Bases \n', matching_bases2)
print('Alice Sifted Key \n', alice_sifted_key)
print('Bob Sifted Key \n', bob_sifted_key2)
print('Corrected Key \n', corrected_key2)
print('Final Sifted Key \n', final_sifted_key2)

