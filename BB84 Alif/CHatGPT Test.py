from qiskit import QuantumCircuit,Aer,transpile
# from qiskit.visualization import plot_bloch_multivector,plot_circuit_layout,plot_histogram
from numpy.random import randint
import numpy as np
import matplotlib.pyplot as mpl
import seaborn as sns 
import time
import math

np.random.seed(seed=1)
#Generate Random Bits
def generate_random_bits(N):
    return [randint(2) for _ in range(N)]

#Generate Random Bases
def generate_random_bases(N):
    return [randint(2) for _ in range(N)]

#Encode bits using bases 
def encode_message(bits, bases):
    qc = QuantumCircuit(len(bits), len(bits))
    for q, (bit, base) in enumerate(zip(bits, bases)):
        if base == 0:
            qc.x(q) if bit == 1 else None
        else:
            qc.h(q)
            qc.x(q) if bit == 1 else None
            qc.h(q)
    qc.barrier()
    return qc

#Measure bits using bases
def measure_message(qc, bases):
    backend = Aer.get_backend('aer_simulator')
    measurements = []
    for q, base in enumerate(bases):
        if base == 0:
            qc.measure(q, q)
        else:
            qc.h(q)
            qc.measure(q, q)
            qc.h(q)
    result = backend.run(qc, shots=1, memory=True).result()
    measurements = [int(bit) for bit in result.get_memory()]
    return measurements 

#Remove unused bits that do not match from alice and bobs bases
def remove_garbage(a_bases, b_bases, bits):
    return [bit for bit, a_base, b_base in zip(bits, a_bases, b_bases) if a_base == b_base]

def check_key(a_key, b_key):
    return sum([a == b for a, b in zip(a_key, b_key)])
        


def count_errors(a_bits, b_bits, e_bits):
    return sum([a != b and e == 1 for a, b,

def qber_calculations(a_key,b_key):
    num_mismatch=sum([1 for i in range(len(a_key)) if a_key[i] != b_key[i]])
    qber=num_mismatch/len(a_key)
    # print('key lenght',len(a_key))
    # print('Mismatch',num_mismatch)
    # print('Qber',qber)
    return(qber)
    
# def channel_capacity_calculations(a_bits,b_bits,b,f,q):
#     H=-q*math.log(q,2)-(1-q)*math.log(1-q,2)
#     return b*f*(1-H)


### RUN Protocol###

error_rates=[]
number_bits=[]
times=[]
qbers=[]
bandwidths=[]

for i in range (5,100):
    start_time=time.time()
    N=i
    #Set Alice Bits and Bases
    alice_bits=generate_random_bits(N)
    alice_bases=generate_random_bases(N)

    #Encode message
    message=encode_message(alice_bits,alice_bases)

    #Get Eve bases and intercept message
    eve_bases=generate_random_bases(N)
    eve_bits=measure_message(message,eve_bases)

    #Generate Bob bases and measure enconded message
    bob_bases=generate_random_bases(N)
    bob_bits=measure_message(message,bob_bases)


    #Remove undesired bits 
    alice_key=remove_garbage(alice_bases,bob_bases,alice_bits)
    bob_key=remove_garbage(alice_bases,bob_bases,bob_bits)

    end_time=time.time()
    
    total_time= end_time-start_time
    times.append(total_time)
    bandwidth=round(len(alice_bits) / total_time)
    # print(bandwidth)
    
    #Measure bandwidth
    bandwidths.append(bandwidth)
    
    #Count Qber
    qber=qber_calculations(alice_key,bob_key)
    qbers.append(qber)
    
    #Measure Channel Capacity
    # succesful_transmissions=check_key(alice_key,bob_key)
    # channel_capacity=channel_capacity_calculations(alice_bits,bob_bits,bandwidth,succesful_transmissions,qber)
    # print(channel_capacity)
    
    #Count Errors 
    num_errors=count_errors(alice_bits,bob_bits,eve_bits)
    error_rate=round(num_errors/float(N),3)
    error_rates.append(error_rate)
    number_bits.append(i)

    
    
    



#PLOTTING

#Bit FLip Errors
mpl.scatter(error_rates,number_bits,c='c')
mpl.ylabel('Number of bits')
mpl.xlabel('Bit Flip Probability')
mpl.show()

#QBER
mpl.scatter(qbers,number_bits,c='r')
mpl.ylabel('Number of bits')
mpl.xlabel('Qber')
mpl.show()

#Bandwidth
mpl.plot(number_bits,bandwidths,c='c',linewidth=.5)
mpl.xlabel('Number of bits')
mpl.ylabel('Bandwidth (Hz)')
mpl.show()


#Formatting Of Graph
mpl.hist(error_rates,bins=20,alpha=0.5,color='c',density=True,edgecolor='k',linewidth=.5)
mpl.ylabel('Frequency')
mpl.xlabel('Probability of bit flip')
mpl.show()

