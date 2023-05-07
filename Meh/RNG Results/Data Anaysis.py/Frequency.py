import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as norm


ByteFile='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/ByteResults1.csv'
BitFile='C:/Users/alifa/OneDrive/Documents/Universitas Malaya/FYP/Quantum/Meh/RNG Results/Used/BitResults1.csv'
ByteResults = pd.read_csv(ByteFile,dtype=str)
BitResults = pd.read_csv(BitFile,dtype=str)

# print(Results.head())
# print(Results.describe())
# print(Results.QRNG.value_counts())

bitstring_order=['00000', '00001', '00010', 
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

colors = {'QRNG': {'color': 'tab:green', 'label': 'QRNG','opacity':1}, 
          'Random Module': {'color': 'tab:red', 'label': 'Random Module','opacity':1}, 
          'IBMQ': {'color': 'tab:blue', 'label': 'IBMQ','opacity':1}}


# Plot frequency distribution of ByteFile
# Plot frequency distribution of ByteFile
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12, 8))
width = 0.2  # the width of the bars
for i, col in enumerate(ByteResults.columns):
    x = np.arange(len(ByteResults[col].value_counts().index))
    counts = ByteResults[col].value_counts().sort_index()
    ax.bar(x - width*(len(ByteResults.columns)-1)/2 + width*i
           , counts, width, alpha=colors.get(col)['opacity']
           , color=colors.get(col)['color'], edgecolor='white', linewidth=1, label=colors.get(col)['label'])
plt.title("Counts of Genrated Bitstring(n=5)")
plt.xlabel('Bitstring')
plt.ylabel('Counts')
plt.xticks(np.arange(len(ByteResults['QRNG'].value_counts().index)), ByteResults['QRNG'].value_counts().sort_index().index,rotation=275)
plt.legend(title='Source', labels=[colors.get(col)['label'] for col in ByteResults.columns], loc='best')
plt.show()

# Create a stacked bar chart for the bit file
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(10, 6))
for i, col in enumerate(BitResults.columns):
    sns.countplot(x=col, data=BitResults, alpha=colors.get(col)['opacity'], 
                  ax=ax, color=colors.get(col)['color'], 
                  edgecolor='white',
                  label=colors.get(col)['label'], zorder=i)
    # Adjust the position of each bar in the stack
    for j, patch in enumerate(ax.patches):
        if j % len(BitResults.columns) == i:
            patch.set_x(patch.get_x() + i * 0.15 - 0.25)

plt.title("Counts of Bits Generated")
plt.xlabel('Bits')
plt.ylabel('Count')
sns.despine()
plt.legend(title='Source')
plt.show()
