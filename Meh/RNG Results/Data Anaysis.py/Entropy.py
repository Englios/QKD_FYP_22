import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as norm

def shannon_entropy(data):
    _, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

colors = {'QRNG': {'color': 'tab:green', 'label': 'QRNG','opacity':1}, 
          'Random Module': {'color': 'tab:red', 'label': 'Random Module','opacity':1}, 
          'IBMQ': {'color': 'tab:blue', 'label': 'IBMQ','opacity':1}}

colors2 = {'QRNG Bytes': {'color': 'tab:green', 'label': 'QRNG','opacity':1}, 
          'Random Module Bytes': {'color': 'tab:red', 'label': 'Random Module','opacity':1}, 
          'IBMQ Bytes': {'color': 'tab:blue', 'label': 'IBMQ','opacity':1}}

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

# Calculate entropy values
entropy_values = data.apply(shannon_entropy)

# Create a scatter plot of entropy values
plt.scatter(x=entropy_values['QRNG'], y=entropy_values['Random Module'], color=colors['Random Module']['color'], label=colors['Random Module']['label'])
plt.scatter(x=entropy_values['QRNG'], y=entropy_values['IBMQ'], color=colors['IBMQ']['color'], label=colors['IBMQ']['label'])
plt.scatter(x=entropy_values['QRNG'], y=entropy_values['QRNG'], color=colors['QRNG']['color'], label=colors['QRNG']['label'])
plt.xlabel('QRNG Entropy')
plt.ylabel('Entropy')
plt.title('Shannon Entropy Analysis')
plt.legend()
plt.show()






