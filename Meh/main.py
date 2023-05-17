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

            noise = np.random.normal(0, self.noise_level, size=noisy_qubit.shape)
            noisy_qubit = np.add(noisy_qubit, noise)

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

"""# Define the parameter values to loop over"""
n_runs = 10
distances = [10, 20, 30, 40, 50,60,70,80,90,100]
bit_length = [50, 100, 150, 200, 250,300,350,400,450,500]
n=10
distance=1000


"""# Initialize arrays to store the data"""
error_rates = []
data_loss_rates = []
final_key_lengths = []

"""Arrays to store data of keys"""
matching_bases = []
matching_indices = []
bob_sifted_key=[]
alice_sifted_key=[]
final_sifted_key=[]

# print('Run: ', run)
for distance in distances:
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


    for i in range(len(alice_bits)):
        if alice_bases[i] == bob_bases[i]:
            matching_bases.append(alice_bases[i])
            matching_indices.append(i)
            alice_sifted_key.append(alice_bits[i])
            bob_sifted_key.append(bob_bits[i])
            if bob_bits[i]== alice_bits[i]:
                final_sifted_key.append(alice_bits[i])

    data_loss=(n-len(bob_sifted_key))/n    
    error_rate=(len(bob_sifted_key)-len(final_sifted_key))/(len(bob_sifted_key))
    
    # """Key Printing """
    # print('Alice Bits \n',alice_bits) 
    # print('Alice Bases \n',alice_bases)    
    # print('Bob Bits \n',bob_bits)
    # print('Bob Bases \n',bob_bases)
    # print('Matching Bases \n',matching_bases)
    # print('Alice Sifted Key \n',alice_sifted_key)       
    # print('Bob Sifted Key \n',bob_sifted_key)
    # print('Sifted Key Lenght \n',len(matching_indices))  
    # print('Final Key \n',final_sifted_key)
    # print('Final Key Lenght \n',len(final_sifted_key)) 
    # print('Data Loss Rate \n',data_loss)
    # print('Error Rate\n',error_rate) 

    data_loss_rates.append(data_loss)   
    error_rates.append(error_rate)

print(data_loss_rates)
print(error_rates)


plt.plot(distances,data_loss_rates,label="Data Loss Rates")
plt.xlabel("Distances (km)")
plt.ylabel("Data Loss (bytes)")
plt.title("Data Loss")
plt.legend()
plt.show()

plt.plot(distances,error_rates,label="Error Rates")
plt.xlabel("Distances (km)")
plt.ylabel("Errors")
plt.title("Error Rates")
plt.legend()
plt.show()

