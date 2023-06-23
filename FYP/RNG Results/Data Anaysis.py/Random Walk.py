import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def parking_lot_test(sequence):
    directions = []  # Initialize an empty list to store the directions

    for i in range(0, len(sequence), 2):
        # Convert pairs of bits to directions
        if sequence[i] == 0 and sequence[i + 1] == 0:
            directions.append(0)  # Move up
        elif sequence[i] == 0 and sequence[i + 1] == 1:
            directions.append(1)  # Move down
        elif sequence[i] == 1 and sequence[i + 1] == 0:
            directions.append(2)  # Move right
        elif sequence[i] == 1 and sequence[i + 1] == 1:
            directions.append(3)  # Move left

    position = [0, 0]  # Initial position of the car
    positions = [position]  # Store the positions of the car

    for direction in directions:
        if direction == 0:  # Move up
            position[0] += 1
        elif direction == 1:  # Move down
            position[0] -= 1
        elif direction == 2:  # Move right
            position[1] += 1
        elif direction == 3:  # Move left
            position[1] -= 1

        positions.append(list(position))  # Store the new position

    positions = np.array(positions)

    return positions


# Compute the car's movements for each sequence
qrng_positions = parking_lot_test(qrng)
random_positions = parking_lot_test(random)
ibmq_positions = parking_lot_test(ibmq)
secrets_positions = parking_lot_test(secrets)

# Calculate the mean positions
qrng_mean = np.mean(qrng_positions, axis=0)
random_mean = np.mean(random_positions, axis=0)
ibmq_mean = np.mean(ibmq_positions, axis=0)
secrets_mean = np.mean(secrets_positions, axis=0)


# Calculate the variances for each sequence
qrng_variance = np.var(qrng_positions)
random_variance = np.var(random_positions)
ibmq_variance = np.var(ibmq_positions)
secrets_variance = np.var(secrets_positions)

# Create a list of sequence names and variances
sequences = ['QRNG', 'Random Module', 'IBMQ', 'Secrets']
variances = [qrng_variance, random_variance, ibmq_variance, secrets_variance]

# Plot the variances
plt.bar(sequences, variances)
plt.xlabel('Random Walk Sequence')
plt.ylabel('Variance')
plt.title('Variance of Random Walk Sequences')
plt.show()
