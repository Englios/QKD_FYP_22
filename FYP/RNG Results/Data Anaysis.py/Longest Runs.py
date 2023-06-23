import numpy as np
from scipy.stats import norm
import pandas as pd

BitFile1 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/ByteResults1.csv'
BitFile2 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/ByteResults2.csv'
BitFile3 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/ByteResults3.csv'
BitFile4 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/ByteResults4.csv'
SecretsFile1 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/SecretByteResults.csv'

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


def longest_run_of_ones(sequence):
    n = len(sequence)
    max_run = 0
    current_run = 0

    for i in range(n):
        if sequence[i] == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    # Calculate the expected value and standard deviation
    expected_value = (2 * n - 1) / 3
    standard_deviation = (16 * n - 29) / 90

    # Calculate the test statistic
    test_statistic = (max_run - expected_value) / standard_deviation

    # Compute the p-value using the standard normal distribution
    p_value = 1 - norm.cdf(test_statistic)

    # Compare the p-value with the significance level
    significance_level = 0.01
    if p_value < significance_level:
        result = "Fail: The sequence may not be sufficiently random."
    else:
        result = "Pass: The sequence is likely random."

    return result


print("Longest Run Test:")
print("QRNG:", longest_run_of_ones(qrng))
print("Random Module:", longest_run_of_ones(random))
print("IBMQ:", longest_run_of_ones(ibmq))
print("Secrets:", longest_run_of_ones(secrets))
