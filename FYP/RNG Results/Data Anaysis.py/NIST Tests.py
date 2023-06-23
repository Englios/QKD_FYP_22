import numpy as np
import pandas as pd
import math

def entropy(sequence):
    n = len(sequence)
    unique_elements, counts = np.unique(sequence, return_counts=True)
    probabilities = counts / n
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return entropy_value

def frequency_test(sequence):
    n = len(sequence)
    ones_count = np.count_nonzero(sequence == 1)
    zeros_count = n - ones_count
    p_value = np.abs(ones_count - zeros_count) / np.sqrt(n)
    p_value = min(p_value, 1.0)  # Cap p-value at 1.0
    passed = (p_value >= 0.01)
    return {"p-value": p_value, "pass": passed}

def monobit_test(sequence):
    n = len(sequence)
    ones_count = np.count_nonzero(sequence == 1)
    s = (ones_count - (n / 2)) / np.sqrt(n / 4)
    p_value = math.erfc(np.abs(s) / np.sqrt(2))
    p_value = min(p_value, 1.0)  # Cap p-value at 1.0
    passed = (p_value >= 0.01)
    return {"p-value": p_value, "pass": passed}

def longest_runs_test(sequence):
    n = len(sequence)
    ones_count = np.count_nonzero(sequence == 1)
    zeros_count = n - ones_count
    rl = np.zeros(n)
    rl[sequence == 1] = -1
    rl[sequence == 0] = 1
    max_run = 0
    current_run = 0
    for i in range(n):
        if rl[i] == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    p_value = math.erfc((max_run - (2 * ones_count * zeros_count) / n) / (2 * np.sqrt(2 * ones_count * zeros_count * (2 * ones_count * zeros_count - n) / (n**2 * (n - 1)))))
    p_value = min(p_value, 1.0)  # Cap p-value at 1.0
    passed = (p_value >= 0.01)
    return {"p-value": p_value, "pass": passed}

def matrix_rank_test(sequence):
    n = len(sequence)
    m = int(n / 32)
    matrix = sequence[:m*32].reshape(m, 32)
    rank = np.linalg.matrix_rank(matrix)
    p_value = math.erfc((rank - (m - 0.5)) / (np.sqrt(2 * m * (m + 1))))
    p_value = min(p_value, 1.0)  # Cap p-value at 1.0
    passed = (p_value >= 0.01)
    return {"p-value": p_value, "pass": passed}

# Read the data from CSV files
BitFile1 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/BitResults1.csv'
BitFile2 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/BitResults2.csv'
BitFile3 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/BitResults3.csv'
BitFile4 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/BitResults4.csv'
SecretsFile1 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/SecretBitResults.csv'

BitResults1 = pd.read_csv(BitFile1, dtype=str)
BitResults2 = pd.read_csv(BitFile2, dtype=str)
BitResults3 = pd.read_csv(BitFile3, dtype=str)
BitResults4 = pd.read_csv(BitFile4, dtype=str)
SecretResults1 = pd.read_csv(SecretsFile1, dtype=str)

bitstring_df = pd.concat([BitResults1, BitResults2, BitResults3, BitResults4])
bitstring_df['Secrets'] = SecretResults1['Secrets']

qrng = np.array(bitstring_df['QRNG'].astype(int))
random = np.array(bitstring_df['Random Module'].astype(int))
ibmq = np.array(bitstring_df['IBMQ'].astype(int))
secrets = np.array(bitstring_df['Secrets'].astype(int))

# Perform statistical tests
qrng_entropy = entropy(qrng)
random_entropy = entropy(random)
ibmq_entropy = entropy(ibmq)
secrets_entropy = entropy(secrets)

qrng_freq_result = frequency_test(qrng)
random_freq_result = frequency_test(random)
ibmq_freq_result = frequency_test(ibmq)
secrets_freq_result = frequency_test(secrets)

qrng_monobit_result = monobit_test(qrng)
random_monobit_result = monobit_test(random)
ibmq_monobit_result = monobit_test(ibmq)
secrets_monobit_result = monobit_test(secrets)

qrng_longest_runs_result = longest_runs_test(qrng)
random_longest_runs_result = longest_runs_test(random)
ibmq_longest_runs_result = longest_runs_test(ibmq)
secrets_longest_runs_result = longest_runs_test(secrets)

qrng_matrix_rank_result = matrix_rank_test(qrng)
random_matrix_rank_result = matrix_rank_test(random)
ibmq_matrix_rank_result = matrix_rank_test(ibmq)
secrets_matrix_rank_result = matrix_rank_test(secrets)

# Display the results
print("Entropy:")
print("QRNG Sequence:", qrng_entropy)
print("Random Module Sequence:", random_entropy)
print("IBMQ Sequence:", ibmq_entropy)
print("Secrets Sequence:", secrets_entropy)

print("\nFrequency Test:")
print("QRNG Sequence:", qrng_freq_result)
print("Random Module Sequence:", random_freq_result)
print("IBMQ Sequence:", ibmq_freq_result)
print("Secrets Sequence:", secrets_freq_result)

print("\nMonobit Test:")
print("QRNG Sequence:", qrng_monobit_result)
print("Random Module Sequence:", random_monobit_result)
print("IBMQ Sequence:", ibmq_monobit_result)
print("Secrets Sequence:", secrets_monobit_result)

print("\nLongest Runs Test:")
print("QRNG Sequence:", qrng_longest_runs_result)
print("Random Module Sequence:", random_longest_runs_result)
print("IBMQ Sequence:", ibmq_longest_runs_result)
print("Secrets Sequence:", secrets_longest_runs_result)

print("\nMatrix Rank Test:")
print("QRNG Sequence:", qrng_matrix_rank_result)
print("Random Module Sequence:", random_matrix_rank_result)
print("IBMQ Sequence:", ibmq_matrix_rank_result)
print("Secrets Sequence:", secrets_matrix_rank_result)

results = [
    ["", "Entropy", "Frequency Test", "Monobit Test", "Longest Runs Test", "Matrix Rank Test"],
    ["QRNG Sequence", qrng_entropy, qrng_freq_result, qrng_monobit_result, qrng_longest_runs_result, qrng_matrix_rank_result],
    ["Random Module Sequence", random_entropy, random_freq_result, random_monobit_result, random_longest_runs_result, random_matrix_rank_result],
    ["IBMQ Sequence", ibmq_entropy, ibmq_freq_result, ibmq_monobit_result, ibmq_longest_runs_result, ibmq_matrix_rank_result],
    ["Secrets Sequence", secrets_entropy, secrets_freq_result, secrets_monobit_result, secrets_longest_runs_result, secrets_matrix_rank_result]
]

# Display the results table
print("Results:")
for row in results:
    print(row)