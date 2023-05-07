import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest

BitFile1='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults1.csv'
BitFile2='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults2.csv'
BitFile3='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults3.csv'
BitFile4='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults4.csv'

BitResults1 = pd.read_csv(BitFile1,dtype=str)
BitResults2 = pd.read_csv(BitFile2,dtype=str)
BitResults3 = pd.read_csv(BitFile3,dtype=str)
BitResults4 = pd.read_csv(BitFile4,dtype=str)


bitstring_df=pd.concat([BitResults1, BitResults2,BitResults3,BitResults4])

# print(Results.head())
# print(Results.describe())
# print(Results.QRNG.value_counts())


colors = {'QRNG': {'color': 'tab:green', 'label': 'QRNG','opacity':1}, 
          'Random Module': {'color': 'tab:red', 'label': 'Random Module','opacity':1}, 
          'IBMQ': {'color': 'tab:blue', 'label': 'IBMQ','opacity':1}}

colors2 = {'QRNG Bytes': {'color': 'tab:green', 'label': 'QRNG','opacity':1}, 
          'Random Module Bytes': {'color': 'tab:red', 'label': 'Random Module','opacity':1}, 
          'IBMQ Bytes': {'color': 'tab:blue', 'label': 'IBMQ','opacity':1}}



# Convert each column to a numpy array of binary values
qrng = np.array(bitstring_df['QRNG'].astype(int))
random = np.array(bitstring_df['Random Module'].astype(int))
ibmq = np.array(bitstring_df['IBMQ'].astype(int))

# Pack the bits into bytes
qrng_bytes = np.packbits(qrng).tolist()
random_bytes = np.packbits(random).tolist()
ibmq_bytes = np.packbits(ibmq).tolist()

# Create a new DataFrame with the byte data
byte_data = pd.DataFrame({
    'QRNG Bytes': qrng_bytes,
    'Random Module Bytes': random_bytes,
    'IBMQ Bytes': ibmq_bytes,
})
print(byte_data)
data=pd.DataFrame(
    {
    'QRNG': qrng,
    'Random Module': random,
    'IBMQ': ibmq,  
    }
)

# # Byte frequency analysis
# byte_counts = byte_data.apply(pd.Series.value_counts)
# byte_freq = byte_counts / byte_counts.sum()

# # Create a new figure with subplots for each byte data set
# fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

# # Loop through each column and plot the KDE
# for i, col in enumerate(byte_data.columns):
#     sns.kdeplot(data=byte_data[col], ax=axes[i], color=colors2[col]['color'], alpha=0.5)
#     axes[i].set_title(colors2[col]['label'])

# # Set the overall title
# fig.suptitle('Byte Distribution KDE')

# plt.show()

# Count the frequency of each byte value for each column
byte_counts = byte_data.apply(pd.value_counts).fillna(0)

# Calculate the frequency of each byte value for each column
byte_freq = byte_counts / byte_data.shape[0]

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, col in enumerate(byte_data.columns):
    ax = axs[i]
    ax.scatter(byte_freq.index, byte_freq[col], alpha=0.5, label=colors2[col]['label'], color=colors2[col]['color'])
    ax.set_xlabel('Byte Value')
    ax.set_ylabel('Frequency')
    ax.set_title(colors2[col]['label'])

plt.suptitle('Byte Frequency by RNG Method', fontsize=16)
plt.show()


# Create subplots
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

# Plot scatter plots on each subplot
for i, col in enumerate(byte_data.columns):
    axs[i].scatter(byte_freq.index, byte_freq[col], alpha=0.5, label=colors2[col]['label'], color=colors2[col]['color'])
    # axs[i].plot([0, 255], [1/256, 1/256], '--k', label='Randomness Threshold')
    axs[i].set_xlabel('Byte Value')
    axs[i].set_ylabel('Frequency')
    axs[i].set_title('Byte Frequency for {}'.format(colors2[col]['label']))

plt.tight_layout()
plt.show()




