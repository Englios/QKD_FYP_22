from qiskit import QuantumCircuit,Aer,transpile
from qiskit.visualization import plot_bloch_multivector,plot_circuit_layout,plot_histogram
from numpy.random import randint
import numpy as np
import matplotlib.pyplot as mpl


# np.random.seed(seed=1)

class qubit():
    def generate_random_bits(N):
        arr = list()
        for i in range(N):
            arr.append(randint(2))
        return arr

    def generate_random_bases(N):
        states=qubit.generate_random_bits(N)
        # i=0
        # while i<N:
        #     if states[i]== 0:
        #         states[i]= 'X'
        #     else:
        #         states[i]='Z'
        #     i+=1
                
        return states

    
def encode_message(bits,bases):
    message=[]
    for i in range(N):
        qc=QuantumCircuit(1,1)
        if bases[i]==0:
            if bits[i]==0:
                pass
            else:
                qc.x(0)
        else:
            if bits[i]==0:
                qc.h(0)
            else:
                qc.x(0)
                qc.h(0)
        qc.barrier()
        message.append(qc)
    return message

def measure_message(message,bases):
    backend=Aer.get_backend('aer_simulator')
    measurements=[]
    for q in range(N):
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

def remove_garbage(a_bases,b_bases,bits):
    good_bits=[]
    for q in range(N):
        if a_bases[q]==b_bases[q]:
            good_bits.append(bits[q])
    return good_bits

def sample_bits(bits,selection):
    sample=[]
    for i in selection:
        i=np.mod(i,len(bits))
        sample.append(bits.pop(i))
    return sample

N=100
alicebits=qubit.generate_random_bits(N)
alicebases=qubit.generate_random_bases(N)
message=encode_message(alicebits,alicebases)
# for i in alicebits:
#     message[i].draw(output='mpl',filename='Test Circuit')


print("Alice bits \n", alicebases)
print("Alice bases \n", alicebits)

bobbases=qubit.generate_random_bases(N)
print('Bob bases\n',bobbases)
bobresults=measure_message(message,bobbases)
print('Bob Results\n',bobresults)

alice_key=remove_garbage(alicebases,bobbases,alicebits)
bob_key=remove_garbage(alicebases,bobbases,bobresults)
print('Alice Key\n',alice_key)
print('Bob Key\n',bob_key)

#Sample

sample_size= round(len(bob_key)/2)
bit_selection= randint(N,size=sample_size)

bob_sample=sample_bits(bob_key,bit_selection)
alice_sample=sample_bits(alice_key,bit_selection)

print('Bob Sample \n' + str(bob_sample))

print('Alice Sample \n'+ str(alice_sample))

print(bool(bob_sample==alice_sample))

message[0].draw(output='mpl',filename='Test Circuit')
mpl.show()