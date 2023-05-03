from QRNG import get_simulated_bits
from qiskit import *
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

random.seed=1

def shannon_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

max_bit_length = 100
random_entropies=[]
qrng_entropies=[]

for bit_lenght in range(1,max_bit_length+1):
    random_bits=[random.randint(0,1) for i in range(bit_lenght)]
    # print("Random Bits:",random_bits)

    _,qrng_bits=get_simulated_bits(bit_lenght,1)
    qrng_bits = [bit for bits in qrng_bits for bit in bits]
    # print("QRNG Bits:",qrng_bits)


    entropy_bits=shannon_entropy(random_bits)
    entropy_qrng=shannon_entropy(qrng_bits)

    # print("Entropy of random module:",entropy_bits)
    # print("Entropy of QRNG:",entropy_qrng)
    
    random_entropies.append(entropy_bits)
    qrng_entropies.append(entropy_qrng)

plt.plot(range(1, max_bit_length+1), random_entropies, label="Random")
plt.plot(range(1, max_bit_length+1), qrng_entropies, label="QRNG")
plt.legend()
plt.xlabel("Bit Length")
plt.ylabel("Shannon Entropy")
plt.title("Entropies of Random Number Generators")
plt.show()

# Kolmogorov-Smirnov test
D, p = ks_2samp(random_entropies, qrng_entropies)
print("KS test statistic:", D)
print("p-value:", p)

#Autocorrelation
# autocorr = np.correlate(qrng_entropies, range(1, max_bit_length+1) , mode='full')
# autocorr = autocorr[autocorr.size // 2:]
# autocorr1 = np.correlate(random_entropies, range(1, max_bit_length+1) , mode='full')
# autocorr1 = autocorr[autocorr1.size // 2:]
# plt.plot(autocorr,label="QRNG",color="orange")
# plt.plot(autocorr1,label="Random",color="blue")
# plt.xlabel('Lag')
# plt.ylabel('Correlation')
# plt.legend()
# plt.show()

