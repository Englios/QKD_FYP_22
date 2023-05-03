from QRNG import get_simulated_bits,get_real_bits,get_rng
from qiskit import *
import numpy as np

def counter(data):
    counts=dict()
    for bit in data:
        if bit not in counts:
            counts[bit] = 1
        else:
            counts[bit] += 1
    return counts

bit_length = 8192


simulated_bytes, simulated_bits = get_simulated_bits(5,bit_length)  # generate 10x10 2D array of random bits
flat_s_bits = np.array(simulated_bits).flatten()
sim_counts = counter(flat_s_bits)  # dictionary to store counts of each bit value
simulated_counts = dict(zip(*np.unique(simulated_bytes, return_counts=True,axis=None)))

simulated_bits = bytearray(np.packbits(simulated_bits))

# Save results in a text file
with open('C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/QRNG Simulator/Simresults2.txt', 'w') as f:
    f.write(f'Simulated Bits: {flat_s_bits}\n')
    f.write(f'Simulated Counts: {sim_counts}\n')
    f.write(f'Byte Array: {simulated_bits}\n')
    f.write(f'Simulated Unique Bit: {simulated_bytes}\n')
    f.write(f'Simulated Unique Counts: {simulated_counts}\n')
    
rng_bytes, rng_bits = get_rng(5,bit_length)  # generate 10x10 2D array of random bits
rng_counts = counter(rng_bits)  # dictionary to store counts of each bit value
rngenerator_counts = dict(zip(*np.unique(rng_bytes, return_counts=True,axis=None)))

rng_bits = bytearray(np.packbits(rng_bits))

# Save results in a text file
with open('C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Random Number Generator/RNGresults2.txt', 'w') as f:
    f.write(f'Real Bits: {np.array(np.unpackbits(rng_bits))}\n')
    f.write(f'Real Counts: {rng_counts}\n')
    f.write(f'Byte Array: {rng_bits}\n')
    f.write(f'Real Unique Bit: {rng_bytes}\n')
    f.write(f'Real Unique Counts: {rngenerator_counts}\n')
    
    
real_bytes, real_bits = get_real_bits(5,bit_length)  # generate 10x10 2D array of random bits
flat_r_bits = np.array(real_bits).flatten()
r_counts = counter(flat_r_bits)  # dictionary to store counts of each bit value
real_counts = dict(zip(*np.unique(real_bytes, return_counts=True,axis=None)))

real_bits = bytearray(np.packbits(real_bits))

# Save results in a text file
with open('C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/IBMQ/IBMQresults2.txt', 'w') as f:
    f.write(f'Real Bits: {flat_r_bits}\n')
    f.write(f'Real Counts: {r_counts}\n')
    f.write(f'Byte Array: {real_bits}\n')
    f.write(f'Real Unique Bit: {real_bytes}\n')
    f.write(f'Real Unique Counts: {real_counts}\n')
    


