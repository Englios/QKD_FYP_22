import numpy as np
import random

n=10

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


class QuantumChanenel:
    def __init__(self,noise_level):
        self.noise_level=noise_level
        
    def apply_noise(self,qubit):
        error=np.random.rand()<self.noise_level
        if error:
            pauli_error=random.choice([X,Z])
            noisy_qubit=pauli_error @ qubit
        else:
            noisy_qubit=qubit
            
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
    
    def decode_qubit(self):
        if self.base == "X":
            self.qubit = X @ self.qubit
        if self.base == "Z":
            self.qubit = Z @ self.qubit
        measurement=np.abs(self.qubit[0])**2>0.5
        
        return measurement
    
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
    
    def measure_qubit(self,qubits,bases):
        bob_measurments = [self.decode_qubit(qubit,base) for qubit,base in zip(qubits,bases)]
        return bob_measurments

# np.set_printoptions(formatter={'all':lambda x : f"{x}, "})

# Generate random bit sequence for the message
sender = Sender()
alice_qubits, alice_bases, alice_bits = sender.generate_message()
alice_qubits_arr=np.array([qubit.qubit for qubit in alice_qubits])

#Printing Alice's Information
print(alice_bits)
print(alice_bases)
# print(alice_qubits_arr)

#Transmit Qubits over Quantum Channel
quantum_channel=QuantumChanenel(noise_level=0.1)
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

