import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def shannon_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

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

data=pd.DataFrame(
    {
    'QRNG': qrng,
    'Random Module': random,
    'IBMQ': ibmq,  
    }
)

# Calculate the correlation matrix and p-values
corr_matrix, p_matrix = np.zeros((3,3)), np.zeros((3,3))
for i, col1 in enumerate(byte_data.columns):
    for j, col2 in enumerate(byte_data.columns):
        corr, pval = pearsonr(byte_data[col1], byte_data[col2])
        corr_matrix[i,j] = corr
        p_matrix[i,j] = pval

# Plot the heatmap
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True,
            xticklabels=byte_data.columns, yticklabels=byte_data.columns, ax=ax)
ax.set_title('Correlation Matrix')
plt.show()