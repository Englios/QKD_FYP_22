from qiskit import QuantumCircuit,Aer,transpile
# from qiskit.visualization import plot_bloch_multivector,plot_circuit_layout,plot_histogram
from numpy.random import randint
import numpy as np
import matplotlib.pyplot as mpl
import seaborn as sns 
import time
import math
import pandas as pd

np.random.seed(seed=1)
#Generate Random Bits
def generate_random_bits(N):
    return [randint(2) for _ in range(N)]

#Generate Random Bases
def generate_random_bases(N):
    return [randint(2) for _ in range(N)]

#Encode bits using bases 
def encode_message(bits,bases):
    message = []
    for bit,base in zip(bits,bases):
        qc=QuantumCircuit(1,1)
        qc.x(0) if bit == 1 and base == 0 else None
        qc.h(0) if base == 1 else None
        qc.barrier()
        message.append(qc)
    return message

#Measure bits using bases
def measure_message(message,bases):
    backend=Aer.get_backend('aer_simulator')
    measurements=[]
    for q in range(len(bases)):
        if bases[q]==0:
            message[q].measure(0,0)
        if bases[q]==1:
            message[q].h(0)
            message[q].measure(0,0)
        aer_sim=Aer.get_backend('aer_simulator')
        result=aer_sim.run(message[q],shots=1,memory=True).result()
        measured_bit=int(result.get_memory()[0])
        measurements.append(measured_bit)
    return measurements 

#Remove unused bits that do not match from alice and bobs bases
def remove_garbage(a_bases,b_bases,bits):
    good_bits=[]
    for q in range(len(bits)):
        if a_bases[q]==b_bases[q]:
            good_bits.append(bits[q])
    return good_bits

def check_key(a_key,b_key):
    match_key=0
    for i in range(len(a_key)):
        if a_key[i]==b_key[i]:
            match_key+=1
    return match_key
        


def count_errors(a_bits,b_bits,e_bits):
    num_errors=0
    for i in range(len(a_bits)):
        if a_bits[i] == b_bits[i] != e_bits[i]:
            num_errors+=1
    return num_errors

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
bit_rates=[]

for i in range (5,1000):
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
    bit_rate=round(len(alice_bits) / total_time)
    # print(bandwidth)
    
    #Measure bandwidth
    bit_rates.append(bit_rate)
    
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
mpl.title('BIT FLIP PROBABILITY')
mpl.show()

#QBER
mpl.scatter(qbers,number_bits,c='r')
mpl.ylabel('Number of bits')
mpl.xlabel('Qber')
mpl.title('QUANTUM BIT ERROR (QBER)')
mpl.show()

mpl.plot(qbers,error_rates,c='r')
mpl.ylabel('Number of bits')
mpl.xlabel('Qber')
mpl.title('QBER VS BIT FLIP')
mpl.show()

#Bandwidth
mpl.plot(number_bits,bit_rates,c='c',linewidth=.5)
mpl.xlabel('Number of bits')
mpl.ylabel('Bit rate (bps)')
mpl.title('BIT RATE ')
mpl.show()

# mpl.plot(qbers,bit_rates,c='c',linewidth=.5)
# mpl.plot(error_rates,bit_rates,c='r',linewidth=.5)
# mpl.xlabel('Probability')
# mpl.ylabel('Bit_rate bps')
# mpl.show()


#Formatting Of Graph
mpl.hist(error_rates,bins=20,alpha=0.5,color='c',density=True,edgecolor='k',linewidth=.5)
mpl.ylabel('Frequency')
mpl.xlabel('Probability of bit flip')
mpl.title('HISTOGRM OF BIT FLIP')
mpl.show()

data={
    "Number of Bits":number_bits,
    "Bit Flip Probability":error_rates,
    "QBER":qbers,
    "Bit Rate":bit_rates,
    "Time taken":times
}
df=pd.DataFrame(data)
df.to_csv(r'C:\Users\alifa\OneDrive\Documents\Universitas Malaya\FYP\Quantum\BB84 Alif\DataFrames\Data3.csv',encoding='utf-8',header=True)