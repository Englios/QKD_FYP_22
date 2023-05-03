import numpy as np
import random

n=50

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
    def __init__(self, distance):
        self.distance = distance
        self.max_noise_level = 0.5
        self.noise_level = self.calculate_noise_level()

    def calculate_noise_level(self):
        return self.max_noise_level * np.exp(-self.distance)

    def apply_noise(self, qubit):
        error = np.random.rand() < self.noise_level
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
                qubit = X @ self.qubit
            elif self.base == "Z":
                qubit = Z @ qubit
            
            measurement = (np.abs(qubit)**2 > 0.5).any()
            
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
    
    def measure_qubit(self, qubits, bases):
        measurements = []
        for qubit, base in zip(qubits, bases):
            measurement=Qubit(0,base).decode_qubit(qubit)
            measurements.append(measurement)
        return measurements

# np.set_printoptions(formatter={'all':lambda x : f"{x}, "})
for distance in range(1,100+1):
    print('Distance: ',distance)
    # Generate random bit sequence for the message
    sender = Sender()
    alice_qubits, alice_bases, alice_bits = sender.generate_message()
    alice_qubits_arr=np.array([qubit.qubit for qubit in alice_qubits])

    #Printing Alice's Information
    print(alice_bits)
    print(alice_bases)
    # print(alice_qubits_arr)

    #Transmit Qubits over Quantum Channel
    quantum_channel=QuantumChannel(distance)
    bob_qubits=[quantum_channel.apply_noise(q.qubit) for q in alice_qubits]

    #Transmit the bases over the public channel
    public_channel=PublicChannel()
    bob_bases=[random.choice(bases) for i in range(n)]
    print(bob_bases)

    # Measure the qubits based on the received bases
    receiver = Receiver()
    bob_measurements = receiver.measure_qubit(bob_qubits, bob_bases)
    print(bob_measurements)

    # Determine the matching bits
    matching_bits = [i for i in range(n) if alice_bases[i] == bob_bases[i]]

    # Use the matching bits as the key
    key = [alice_bits[i] for i in matching_bits if alice_bases[i] == bob_bases[i]]

