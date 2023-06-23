from RNGSource import * # import the functions to generate random bits
from qiskit import *
import numpy as np
import csv

def counter(data):
    """
    Function to count the number of occurrences of each bit value in a given array.

    Args:
        data: Array of bits.

    Returns:
        Dictionary containing the counts of each bit value.
    """
    counts = dict()
    for bit in data:
        if bit not in counts:
            counts[bit] = 1
        else:
            counts[bit] += 1
    return counts

bit_length = 8192*4

byte_file='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/SecretByteResults.csv'
bit_file='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/SecretBitResults.csv'

headers=['QRNG', 'Random Module','IBMQ','Secrets']

"""QRNG Simulator"""
simulated_bytes, simulated_bits = get_simulated_bits(5, bit_length)  # generate 10x10 2D array of random bits
rng_bytes, rng_bits = get_rng(5, bit_length)  # generate random bits using the random module
secrets_bytes, secrets_bits = get_secrets(5, bit_length)  # generate random bits using the random module
ibmq_bytes, ibmq_bits = get_ibmq_bits(5, bit_length)  # generate random bits using the IBMQ quantum computer
flat_s_bits = np.array(simulated_bits).flatten()  # flatten the simulated bits array
flat_ibmq_bits = np.array(ibmq_bits).flatten()  # flatten the IBMQ bits array

# Save byte results in a CSV file
with open(byte_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for i in range(bit_length):  
        writer.writerow([simulated_bytes[i], rng_bytes[i],ibmq_bytes[i],secrets_bytes[i]])  # write each row of the array to the CSV file


# Save bit results in a CSV file
with open(bit_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Secrets"])
    for i in range(len(secrets_bits)): 
        writer.writerow([simulated_bits[i], rng_bits[i],ibmq_bits[i],secrets_bits[i]])
        
