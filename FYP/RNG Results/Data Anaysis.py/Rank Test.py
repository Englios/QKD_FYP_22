import numpy as np
from scipy.stats import chisquare
import pandas as pd

BitFile1='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/BitResults1.csv'
BitFile2='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/BitResults2.csv'
BitFile3='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/BitResults3.csv'
BitFile4='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/BitResults4.csv'
SecretsFile1='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/SecretBitResults.csv'

BitResults1 = pd.read_csv(BitFile1,dtype=str)
BitResults2 = pd.read_csv(BitFile2,dtype=str)
BitResults3 = pd.read_csv(BitFile3,dtype=str)
BitResults4 = pd.read_csv(BitFile4,dtype=str)
SecretResults1 = pd.read_csv(SecretsFile1,dtype=str)


bitstring_df=pd.concat([BitResults1, BitResults2,BitResults3,BitResults4])
bitstring_df['Secrets']=SecretResults1['Secrets']

qrng = np.array(bitstring_df['QRNG'].astype(int))
random = np.array(bitstring_df['Random Module'].astype(int))
ibmq = np.array(bitstring_df['IBMQ'].astype(int))
secrets=np.array(bitstring_df['Secrets'].astype(int))

def rank_test(sequence):
    sequence_size = len(sequence)

    # Find suitable dimensions for reshaping
    for m in range(1, sequence_size + 1):
        if sequence_size % m == 0:
            n = sequence_size // m
            break
    else:
        return "Cannot find suitable dimensions for reshaping."

    # Generate the binary matrix from the sequence
    matrix = np.reshape(sequence, (m, n))

    # Compute the rank of the matrix
    rank = np.linalg.matrix_rank(matrix)

    # Calculate the expected rank
    expected_rank = min(m, n) // 2 + 1
    
    print(rank)
    print(expected_rank)
    # Compare the observed rank with the expected rank
    if rank >= expected_rank:
        result = "Pass: The sequence is likely random."
    else:
        result = "Fail: The sequence may not be sufficiently random."

    return result

qrng_result = rank_test(qrng)
print("QRNG:", qrng_result)

random_result = rank_test(random)
print("Random Module:", random_result)

ibmq_result = rank_test(ibmq)
print("IBMQ:", ibmq_result)