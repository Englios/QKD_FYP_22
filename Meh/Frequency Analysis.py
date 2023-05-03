from QRNG import get_simulated_bits,get_real_bits
from qiskit import *
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# Frequency Analysis
##By bit
bit_length = 1000
_, simulated_bits = get_simulated_bits(bit_length, 1)  # generate 10x10 2D array of random bits
_, real_bits = get_real_bits(bit_length,1)

# convert 2D array to 1D array
flat_s_bits = np.array(simulated_bits).flatten()
flat_r_bits =np.array(real_bits).flatten()

sim_counts = dict()  # dictionary to store counts of each bit value
real_counts = dict()

# count occurrences of each bit value
for bit in flat_s_bits:
    if bit not in sim_counts:
        sim_counts[bit] = 1
    else:
        sim_counts[bit] += 1
        
for bit in flat_r_bits:
    if bit not in sim_counts:
        real_counts[bit] = 1
    else:
        real_counts[bit] += 1
        

# plot histogram
bar_width = 0.4  # width of each bar
plt.bar(np.array(list(sim_counts.keys())) - bar_width/2, list(sim_counts.values()), width=bar_width, alpha=0.5, label="Simulated")
plt.bar(np.array(list(real_counts.keys())) + bar_width/2, list(real_counts.values()), width=bar_width, alpha=0.5, label="Real")
plt.xlabel("Bit Value")
plt.ylabel("Count")
plt.title("Frequency Distribution of Bits")
plt.xticks([0, 1])
plt.legend()
plt.show()
plt.savefig("Freuqncy Distirbution of Bit.png")

##By Unique bit values generated from qubits
simulated_bits, _ = get_simulated_bits(5, 1000)
real_bits, _ = get_real_bits(5, 1000)# generate 10x10 2D array of random bits

# count occurrences of each bit value
simulated_counts = dict(zip(*np.unique(simulated_bits, return_counts=True,axis=None)))
real_counts = dict(zip(*np.unique(real_bits, return_counts=True,axis=None)))

# plot histogram
plt.bar(simulated_counts.keys(), simulated_counts.values(), color='blue', alpha=0.5, label='Simulated')
plt.bar(real_counts.keys(), real_counts.values(), color='orange', alpha=0.5, label='Real')
plt.xlabel("Bit Value")
plt.ylabel("Count")
plt.title("Frequency Distribution of Quantum Computer Generated Unique Bits")
plt.xticks(list(simulated_counts.keys()), rotation=270)
plt.legend()
plt.show()
plt.savefig("Frequency Distribution of Quantum Computer Generated Unique Bits.png")