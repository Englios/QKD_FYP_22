import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

File1 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/ByteResults1.csv'
File2 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/ByteResults2.csv'
File3 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/ByteResults3.csv'
File4 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/ByteResults4.csv'
File5 = 'C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/FYP/RNG Results/Used/SecretByteResults.csv'

ByteResults1 = pd.read_csv(File1, dtype=str)
ByteResults2 = pd.read_csv(File2, dtype=str)
ByteResults3 = pd.read_csv(File3, dtype=str)
ByteResults4 = pd.read_csv(File4, dtype=str)
ByteResults5=pd.read_csv(File5, dtype=str)

ByteResults = pd.concat([ByteResults1, ByteResults2, ByteResults3, ByteResults4])
ByteResults['Secrets']=ByteResults5['Secrets']

# print(ByteResults.head())

bitstring_order = ['00000', '00001', '00010',
                   '00011', '00100', '00101',
                   '00110', '00111', '01000',
                   '01001', '01010', '01011',
                   '01100', '01101', '01110',
                   '01111', '10000', '10001',
                   '10010', '10011', '10100',
                   '10101', '10110', '10111',
                   '11000', '11001', '11010',
                   '11011', '11100', '11101',
                   '11110', '11111']

colors = {'QRNG': {'color': '#FF1D00', 'label': 'QRNG', 'opacity': 0.5},
          'Random Module': {'color': '#84BD00', 'label': 'Random Module', 'opacity': 0.5},
          'IBMQ': {'color': '#006AFF', 'label': 'IBMQ', 'opacity': 0.5},
          'Secrets': {'color': 'purple', 'label': 'Secrets', 'opacity': 0.5}
}

# Set the background style
sns.set_style("white")

# Create a figure with three subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 5))

for i, col in enumerate(ByteResults.columns):
    ax = axes[i // 2, i % 2]  # Adjust indexing for subplots
    counts = ByteResults[col].value_counts().reindex(bitstring_order).fillna(0)
    x = np.arange(len(bitstring_order))
    ax.bar(x, counts, alpha=colors.get(col)['opacity'], color=colors.get(col)['color'], edgecolor='white')
    ax.set_title(colors.get(col)['label'])
    ax.set_xlabel('Bitstring')
    ax.set_ylabel('Counts')
    ax.set_xticks(x)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xticklabels(bitstring_order, rotation=275, size=12)
    # ax.legend([colors.get(col)['label']])

    # Calculate variance and add error bars
    variance = np.var(counts)
    ax.errorbar(x, counts, yerr=np.sqrt(variance), fmt='none', color='gray', capsize=3)
    
    # Calculate mean and add mean line
    mean = np.mean(counts)
    ax.axhline(mean, color='red', linestyle='--')
    
# Set the overall title
fig.suptitle("Counts of Generated Bitstring (n=5)", fontsize=16)

plt.tight_layout()
plt.show()
