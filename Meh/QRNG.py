from qiskit import *
from matplotlib import pyplot as mpl
from qiskit_ibm_provider import IBMProvider
import random 
import numpy as np

# IBMProvider.save_account(token='ccf27abe8db1e1fde20675c6026627d0489f92da4e709353349abe4528f0dd255fb351c6d1a0cab894c65a61bab2ee5100909791fcb59e4725d754b555035520')


def get_simulated_bits(bit_lenght,number_bits):
    qr=QuantumRegister(bit_lenght)
    cr=ClassicalRegister(bit_lenght)
    circuit=QuantumCircuit(qr,cr)
       
    circuit.h(qr)
    circuit.measure(qr,cr)
    # circuit.draw(output='mpl',scale=1,filename='Test')
    # mpl.show()

    backend=Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend, shots=number_bits, memory=True).result()    
    rawvalues_sim = result.get_memory()
    
    bits_array = []
    for bits_str in rawvalues_sim:
        bits = [int(bit) for bit in bits_str]
        bits_array.append(bits)
    
    return rawvalues_sim,bits_array

def get_ibmq_bits(bit_lenght,number_bits):
    qr=QuantumRegister(bit_lenght)
    cr=ClassicalRegister(bit_lenght)
    circuit=QuantumCircuit(qr,cr)
    
    circuit.h(qr)
    circuit.measure(qr,cr)
    # circuit.draw(output='mpl',scale=1,filename='Test')
    # mpl.show()

    provider = IBMProvider()
    qbackend = provider.get_backend('ibmq_manila')
    job = execute(circuit, backend=qbackend, shots=number_bits, memory=True)
    rawvalues_ibmq = job.result().get_memory()
    
    bits_array = []
    for bits_str in rawvalues_ibmq:
        bits = [int(bit) for bit in bits_str]
        bits_array.append(bits)
    
    return rawvalues_ibmq,bits_array

def get_rng(bit_length, number_bits):
    bits_array = []
    rawvalues = []
    for i in range(number_bits):
        byte = ''
        for j in range(bit_length):
            bit = random.randint(0, 1)
            byte += str(bit)
            bits_array.append(bit)
        rawvalues.append(byte)
    return rawvalues, bits_array
