import numpy as np
import random
import matplotlib.pyplot as plt


bases=["X","Z"]
#Z basis Matrices
Z0 = 1/np.sqrt(2)*np.array([[1],[1]])
Z1 = 1/np.sqrt(2)*np.array([[1],[-1]])
#Z gate
Z=np.concatenate((Z0,Z1),axis=1)

#X basis Matrices
X0 = np.array([[1],[0]])
X1= np.array([[0],[1]])
#X gate
X=np.concatenate((X0,X1),axis=1)


class QuantumChannel:
    def __init__(self, distance, k=0.1, A=0.5):
        self.distance = distance
        self.error_probability = A * np.exp(-k * distance)
        self.max_noise_level = 1
        self.noise_level = self.calculate_noise_level()

    def calculate_noise_level(self):
        return self.max_noise_level * np.exp(-(self.distance)**2/100)

    def apply_noise(self, qubit):
        error = np.random.rand() < self.error_probability
        if error:
            pauli_error = random.choice([X, Z])
            noisy_qubit = pauli_error @ qubit
        else:
            noisy_qubit = qubit

        return noisy_qubit
    
class PublicChannel:
    def __init__(self) -> None:
        pass
    def transmit(self,message):
        return message
    
class Qubit:
    def __init__(self,bit,base) -> None:
        self.bit = bit
        self.base= base
        self.qubit =self.encode_qubit()
        
    def encode_qubit(self):
        qubit=np.array([1,0])
        if self.bit == 1:
            qubit = X @ qubit
        if self.base == 'Z':
            qubit = Z @ qubit
        return qubit
    
    def decode_qubit(self,qubit):
        try:
            if self.base == "X":
                qubit = X @ qubit
            elif self.base == "Z":
                qubit = Z @ qubit
            
            measurement = np.abs(qubit)**2 > 0.5
            
            return np.any(measurement)
            
        except ValueError:
            pass

class Sender:
    def __init__(self) -> None:
        pass
    
    def generate_message(self):
        alice_bits=[random.randint(0,1) for i in range(n)]
        alice_bases=[random.choice(bases) for i in range(n)]
        alice_qubits=[Qubit(bit,base) for bit,base in zip(alice_bits,alice_bases)]
        
        return alice_qubits,alice_bases,alice_bits
    
class Receiver:
    def __init__(self) -> None:
        pass
    
    def measure_qubits(self, qubits, bases):
        received_qubits = []
        measurements = []
        bob_bases={}
        bob_bits=[]
        for i, (qubit, base) in enumerate(zip(qubits, bases)):
            # measure the qubit in the eve base
            measurement = Qubit(0,base).decode_qubit(qubit)
            measurements.append(measurement)
            
            bob_bases[i] = base
            #encode qubits in eve base
            if bob_bases[i] == base:
                received_qubits.append(Qubit(measurement, base).qubit)
            else:
                received_qubits.append(Qubit(1-measurement, base).qubit)

        for i, measurement in enumerate(measurements):
            if measurement == 1:
                bob_bits.append(alice_bits[i])
            else:
                bob_bits.append(random.randint(0, 1))

        return received_qubits,bob_bases,measurements,bob_bits
    
class Eavesdropper:
    def __init__(self) -> None:
        pass
    
    def intercept_and_forward(self, qubits, bases, alice_bits):
        intercepted_qubits = []
        measurements = []
        eve_bases={}
        eve_bits=[]
        for i, (qubit, base) in enumerate(zip(qubits, bases)):
            # measure the qubit in the eve base
            measurement = Qubit(0,base).decode_qubit(qubit)
            measurements.append(measurement)
            
            eve_bases[i] = base
            #encode qubits in eve base
            if eve_bases[i] == base:
                intercepted_qubits.append(Qubit(measurement, base).qubit)
            else:
                intercepted_qubits.append(Qubit(1-measurement, base).qubit)

        for i, measurement in enumerate(measurements):
            if measurement == 1:
                eve_bits.append(alice_bits[i])
            else:
                eve_bits.append(random.randint(0, 1))

        return intercepted_qubits,eve_bases,measurements,eve_bits


# Initialize arrays to store the data
error_rates = []
data_loss = []
final_key_lengths = []

# Define the parameter values to loop over
n_runs = [np.linspace(1,10,1)]
distances = [10, 20, 30, 40, 50]
bit_lengths = [50, 100, 150, 200, 250]

# Loop over the parameter values
for distance in distances:
    print('Distance: ', distance)
    error_rates_row = []
    data_loss_row = []
    final_key_lengths_row = []
    
    for n in bit_lengths:
        print('Bit Length: ', n)
        error_rates_run = []
        data_loss_run = []
        final_key_lengths_run = []
        
        for run in n_runs:  
            print('Run: ', run)
            """# Generate random bit sequence for the message"""
            sender = Sender()
            alice_qubits, alice_bases, alice_bits = sender.generate_message()

            """#Transmit Qubits over Quantum Channel"""
            quantum_channel=QuantumChannel(distance)
            noisy_qubits=[quantum_channel.apply_noise(q.qubit) for q in alice_qubits]

            alice_qubits=np.array([qubit.qubit for qubit in alice_qubits])

            """#Initialize Eve"""
            eve=Eavesdropper()
            eve_bases=[random.choice(bases) for i in range(n)]
            intercepted_qubits, _, intercepted_measurements, intercepted_bits = eve.intercept_and_forward(noisy_qubits, eve_bases, alice_bits)

            """#Transmit the bases over the public channel"""
            public_channel=PublicChannel()
            bob_bases=[random.choice(bases) for i in range(n)]

            """# Measure the qubits based on the received bases"""
            receiver = Receiver()
            bob_qubits, bob_bases, bob_measurements, bob_bits = receiver.measure_qubits(intercepted_qubits, bob_bases)

            """Create Key"""
            matching_bits = []
            matching_bases = []
            matching_indices = []

            for i in range(len(alice_bits)):
                if alice_bases[i] == bob_bases[i]:
                    matching_bases.append(alice_bases[i])
                    matching_indices.append(i)
                    if alice_bits[i] == bob_bits[i]:
                        matching_bits.append(alice_bits[i])

            """Error"""            
            mismatches = len(bob_bits) - len(matching_bits)
            error_rate = mismatches / len(bob_bits)
            data_loss_rate = len(bob_bits)-len(matching_bits) / len(bob_bits)
            final_key_length = len(matching_bits)
            
            error_rates_run.append(error_rate)
            data_loss_run.append(data_loss_rate)
            final_key_lengths_run.append(final_key_length)
            
        error_rates_row.append(sum(error_rates_run) / n_runs)
        data_loss_row.append(sum(data_loss_run) / n_runs)
        final_key_lengths_row.append(sum(final_key_lengths_run) / n_runs)

    error_rates.append(error_rates_row)
    data_loss.append(data_loss_row)
    final_key_lengths.append(final_key_lengths_row)

# Plot the results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
titles = ['Error Rates', 'Data Loss', 'Final Key Lengths']
data = [error_rates, data_loss, final_key_lengths]

for i in range(len(titles)):
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Bit Length')
    axs[i].set_ylabel('Distance')
    axs[i].set_xticks(np.arange(len(distances)))
    axs[i].set_yticks(np.arange(len(bit_lengths)))
    axs[i].imshow(data[i], cmap='hot')

plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
titles = ['Error Rates', 'Data Loss', 'Final Key Lengths']
data = [error_rates, data_loss, final_key_lengths]

for i in range(len(titles)):
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Distance')
    axs[i].set_ylabel(data[i])
    axs[i].plot(bit_lengths,data[i])

plt.show()