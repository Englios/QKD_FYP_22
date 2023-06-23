import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from matplotlib.lines import Line2D


bases = ["X", "Z"]
# Z basis Matrices
Z0 = 1 / np.sqrt(2) * np.array([[1], [1j]])
Z1 = 1 / np.sqrt(2) * np.array([[1], [-1j]])
# Z gate
Z = np.concatenate((Z0, Z1), axis=1)

# X basis Matrices
X0 = 1 / np.sqrt(2) * np.array([[1], [1]])
X1 = 1 / np.sqrt(2) * np.array([[1], [-1]])
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

        polarization_noise = np.random.choice(
            [0, 1], size=qubit.shape, p=[self.polarization_fidelity, 1 - self.polarization_fidelity]
        )
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
n_runs = 10
distances = np.linspace(0, 1000, 11)
bit_lengths = np.arange(100, 1000, 20)
n=500

attenuation_coefficient = 0.5
phase_noise_std = 0.5
fidelity = 0.1

"""# Initialize arrays to store the data"""
error_rates1 = np.zeros((len(distances), len(bit_lengths)))
error_rates2 = np.zeros((len(distances), len(bit_lengths)))
data_loss_rates1 = np.zeros((len(distances), len(bit_lengths)))
data_loss_rates2 = np.zeros((len(distances), len(bit_lengths)))
key_rates = np.zeros((len(distances), len(bit_lengths)))
final_key_lengths = []

# matching_bases = []
# matching_indices = []
# bob_sifted_key = []
# alice_sifted_key = []
# final_sifted_key = []
# shared_sifted_key = []
for distance_idx, distance in enumerate(distances):
    matching_bases = []
    matching_indices = []
    bob_sifted_key = []
    alice_sifted_key = []
    final_sifted_key = []
    shared_sifted_key = []
    
    """# Generate random bit sequence for the message"""
    sender = Sender(qrng)
    alice_qubits, alice_bases, alice_bits = sender.generate_message(n)

    """# Transmit Qubits over Quantum Channel"""
    quantum_channel = QuantumChannel(distance, attenuation_coefficient, phase_noise_std, fidelity)
    noisy_qubits = [quantum_channel.apply_noise(qubit.qubit) for qubit in alice_qubits]

    alice_qubits = np.array([qubit.qubit for qubit in alice_qubits])

    """# Initialize Eve"""
    eve = Eavesdropper(0.5)
    eve_bases = [random.choice(bases) for _ in range(n)]
    intercepted_qubits, _, intercepted_measurements, intercepted_bits = eve.intercept_and_forward(noisy_qubits, eve_bases, alice_bits)

    """# Transmit the bases over the public channel"""
    public_channel = PublicChannel()
    bob_bases = [random.choice(bases) for _ in range(n)]

    """# Measure the qubits based on the received bases"""
    receiver = Receiver()
    bob_qubits, bob_bases, bob_measurements, bob_bits = receiver.measure_qubits(intercepted_qubits, bob_bases)
    
    """# Sift the bits based on the matching bases"""
    for i in range(len(alice_bits)):
        if alice_bases[i] == bob_bases[i]:
            matching_bases.append(alice_bases[i])
            matching_indices.append(i)
            alice_sifted_key.append(alice_bits[i])
            bob_sifted_key.append(bob_bits[i])
            if bob_bits[i] == alice_bits[i]:
                shared_sifted_key.append(alice_bits[i])
    
    """Error Reconciliation"""            
    error_reconciliation = ErrorReconciliation(0.5)
    parity_bits = error_reconciliation.parity_check(alice_sifted_key, bob_sifted_key)
    corrected_key = error_reconciliation.correct_errors(bob_sifted_key, parity_bits)
    for i in range(len(alice_sifted_key)):
        if corrected_key[i] == alice_sifted_key[i]:
            final_sifted_key.append(corrected_key[i])
            
    """# Calculate the key rates"""
    uncorrected_data_loss = (n - len(shared_sifted_key)) / n
    corrected_data_loss = (n - len(final_sifted_key)) / n
    uncorrected_error_rate = (len(alice_sifted_key)-len(shared_sifted_key)) / len(alice_sifted_key)
    corrected_error_rate = (len(alice_sifted_key)-len(final_sifted_key)) / len(alice_sifted_key)
    
    # print('-------------')   
    # print(len(alice_sifted_key), len(shared_sifted_key), len(final_sifted_key),sep='\n')
    # Calculate key rate
    key_rate = len(final_sifted_key) / n
    key_rates[distance_idx] = key_rate

    data_loss_rates1[distance_idx] = uncorrected_error_rate
    data_loss_rates2[distance_idx] = corrected_error_rate
    error_rates1[distance_idx] = uncorrected_error_rate
    error_rates2[distance_idx] = corrected_error_rate


plt.figure(figsize=(10, 6))

flat_data_loss_rates1 = np.concatenate(data_loss_rates1)
flat_data_loss_rates2 = np.concatenate(data_loss_rates2)
flat_bit_lengths = np.tile(bit_lengths, len(distances))

for distance_idx, distance in enumerate(distances):
    alpha = (distance_idx + 1) / (len(distances) + 1)  # Calculate alpha value for color
    plt.scatter(bit_lengths, data_loss_rates1[distance_idx], color='blue', alpha=alpha, s=8)
    plt.scatter(bit_lengths, data_loss_rates2[distance_idx], color='red', alpha=alpha, s=8)

# Plot the fitted lines
slope, intercept, r_value, p_value, std_err = linregress(flat_bit_lengths, flat_data_loss_rates1)
sns.regplot(x=flat_bit_lengths, y=flat_data_loss_rates1, ci=95, scatter=False, label='Uncorrected Data Losses',
            color='orange')

slope, intercept, r_value, p_value, std_err = linregress(flat_bit_lengths, flat_data_loss_rates2)
sns.regplot(x=flat_bit_lengths, y=flat_data_loss_rates2, ci=95, scatter=False, label='Corrected Data Losses',
            color='purple')

plt.xlabel("Bit Lengths")
plt.ylabel("Data Loss Rates")
plt.title("Data Loss Rates")

# Create separate legend elements for each color
legend_elements1 = []
legend_elements2 = []

legend_elements1.append(Line2D([0], [0],color='orange', markersize=8,label='Uncorrected'))
legend_elements2.append(Line2D([0], [0],color='purple', markersize=8,label='Corrected'))

for distance_idx, distance in enumerate(distances):
    alpha = (distance_idx + 1) / (len(distances) + 1)
    label = f'{int(distance)} km'
    legend_elements1.append(Line2D([0], [0], marker='o', c='w',markerfacecolor='red', markersize=8,
                                  alpha=alpha,label=label))
    legend_elements2.append(Line2D([0], [0], marker='o',c='w',markerfacecolor='blue', markersize=8,
                                  alpha=alpha, label=label))
# Display the legends outside the plot
plt.legend(handles=legend_elements1 + legend_elements2, loc='center left',labelspacing=0.1, 
           bbox_to_anchor=(1, 0.5),borderaxespad=0.2, borderpad=0.8, ncol=2,fontsize='small',
           title='Distances')
plt.tight_layout()  # Adjust the layout to prevent overlapping elements

plt.show()

flat_error_rates1 = np.concatenate(error_rates1)
flat_error_rates2= np.concatenate(error_rates2)

# Plot the original data and the fitted line with predictive area
plt.figure(figsize=(10, 6))
for distance_idx, distance in enumerate(distances):
    alpha=(distance_idx+1)/(len(distances)+1)
    plt.plot(bit_lengths, error_rates1[distance_idx],'o',c='blue',linewidth=1, markersize=3,alpha=alpha)
    plt.plot(bit_lengths, error_rates2[distance_idx],'o',c='red',linewidth=1, markersize=3,alpha=alpha)

# Plot the fitted line
slope, intercept, r_value, p_value, std_err = linregress(flat_bit_lengths, flat_error_rates1)
sns.regplot(x=flat_bit_lengths, y=flat_error_rates1, ci=95, scatter=False, label='Uncorrected Erro Rates',color='orange')

slope, intercept, r_value, p_value, std_err = linregress(flat_bit_lengths, flat_error_rates2)
sns.regplot(x=flat_bit_lengths, y=flat_error_rates2, ci=95, scatter=False, label='Corrected Error Rates',color='purple')

# Adjust the layout to prevent overlapping elements
plt.xlabel("Bit Lengths")
plt.ylabel("Errors")
plt.title("Error Rates with Fitted Line and Predictive Area")


# Create separate legend elements for each color
legend_elements1 = []
legend_elements2 = []

legend_elements1.append(Line2D([0], [0],color='orange', markersize=8,label='Uncorrected'))
legend_elements2.append(Line2D([0], [0],color='purple', markersize=8,label='Corrected'))

for distance_idx, distance in enumerate(distances):
    alpha = (distance_idx + 1) / (len(distances) + 1)
    label = f'{int(distance)} km'
    legend_elements1.append(Line2D([0], [0], marker='o', c='w',markerfacecolor='red', markersize=8,
                                  alpha=alpha,label=label))
    legend_elements2.append(Line2D([0], [0], marker='o',c='w',markerfacecolor='blue', markersize=8,
                                  alpha=alpha, label=label))
# Display the legends outside the plot
plt.legend(handles=legend_elements1 + legend_elements2, loc='center left',labelspacing=0.1, 
           bbox_to_anchor=(1, 0.5),borderaxespad=0.2, borderpad=0.8, ncol=2,fontsize='small',
           title='Distances')

plt.tight_layout()
plt.show()





