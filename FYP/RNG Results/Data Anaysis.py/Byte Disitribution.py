import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


BitFile1='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults1.csv'
BitFile2='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults2.csv'
BitFile3='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults3.csv'
BitFile4='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults4.csv'
SecretsFile1='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/SecretBitResults.csv'

BitResults1 = pd.read_csv(BitFile1,dtype=str)
BitResults2 = pd.read_csv(BitFile2,dtype=str)
BitResults3 = pd.read_csv(BitFile3,dtype=str)
BitResults4 = pd.read_csv(BitFile4,dtype=str)
SecretResults1 = pd.read_csv(SecretsFile1,dtype=str)


bitstring_df=pd.concat([BitResults1, BitResults2,BitResults3,BitResults4])
bitstring_df['Secrets']=SecretResults1['Secrets']

# print(bitstring_df.head())
# print(bitstring_df.describe())
# print(bitstring_df.QRNG.value_counts())

colors = {'QRNG': {'color': '#FF1D00', 'label': 'QRNG', 'opacity': 0.5},
          'Random Module': {'color': '#84BD00', 'label': 'Random Module', 'opacity': 0.5},
          'IBMQ': {'color': '#006AFF', 'label': 'IBMQ', 'opacity': 0.5},
          'Secrets' : {'color': 'purple', 'label': 'Secrets', 'opacity': 0.5}
            }

colors2 = {'QRNG Bytes': {'color':  '#FF1D00', 'label': 'QRNG','opacity':1}, 
          'Random Module Bytes': {'color': '#84BD00', 'label': 'Random Module','opacity':1}, 
          'IBMQ Bytes': {'color': '#006AFF', 'label': 'IBMQ','opacity':1},
          'Secrets Bytes':{ 'color': 'purple', 'label': 'Secrets','opacity':1}
          }



# Convert each column to a numpy array of binary values
qrng = np.array(bitstring_df['QRNG'].astype(int))
random = np.array(bitstring_df['Random Module'].astype(int))
ibmq = np.array(bitstring_df['IBMQ'].astype(int))
secrets=np.array(bitstring_df['Secrets'].astype(int))

# Pack the bits into bytes
qrng_bytes = np.packbits(qrng).tolist()
random_bytes = np.packbits(random).tolist()
ibmq_bytes = np.packbits(ibmq).tolist()
secrets_bytes = np.packbits(secrets).tolist()

# Create a new DataFrame with the byte data
byte_data = pd.DataFrame({
    'QRNG Bytes': qrng_bytes,
    'Random Module Bytes': random_bytes,
    'IBMQ Bytes': ibmq_bytes,
    'Secrets Bytes': secrets_bytes
})
# print(byte_data)
data=pd.DataFrame(
    {
    'QRNG': qrng,
    'Random Module': random,
    'IBMQ': ibmq,
    'Secrets': secrets  
    }
)

# Byte frequency analysis
byte_counts = byte_data.apply(pd.Series.value_counts)
byte_freq = byte_counts / byte_counts.sum()

# Create a new figure with subplots for each byte data set
fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)

# Loop through each column and plot the KDE
for i, col in enumerate(byte_data.columns):
    sns.kdeplot(data=byte_data[col], ax=axes[i], color=colors2[col]['color'], alpha=0.5)
    axes[i].set_title(colors2[col]['label'])

# Set the overall title
fig.suptitle('Byte Distribution KDE')

plt.show()

# Calculate the mean count for each column
byte_mean_count = byte_counts.mean()

# Calculate the variance of the counts for each column
byte_var_count = byte_counts.var()

# Create subplots
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15,5))

# Plot scatter plots on each subplot
for i, col in enumerate(byte_data.columns):
    axs[i].scatter(byte_counts.index, byte_counts[col], alpha=0.5, label=colors2[col]['label'], color=colors2[col]['color'])
    
    # Plot variance of counts as a filled square
    axs[i].fill_between(byte_counts.index, byte_mean_count[col] - np.sqrt(byte_var_count[col]), byte_mean_count[col] + np.sqrt(byte_var_count[col]),
                        alpha=0.25, color='orange', label='Variance')
    
    # Plot mean count as a line
    axs[i].plot(byte_counts.index, np.full_like(byte_counts.index, byte_mean_count[col]), 'r--', label='Mean Count')
    
    axs[i].set_xlabel('Byte Value')
    axs[i].set_ylabel('Frequency')
    axs[i].set_ylim(20,160)
    axs[i].set_title('Byte Frequency for {}'.format(colors2[col]['label']))
    axs[i].legend()

plt.tight_layout()
plt.show()




