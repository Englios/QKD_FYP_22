import matplotlib.pyplot as mp
import matplotlib.pyplot as plt
import numpy as np

# Values for SiO2
PureA = [0.696749, 0.408218, 0.890815]
PureL = [0.069066, 0.115662, 9.900559]

# Values for Mixture of Si02 and Ge02
MixedA = [0.71104, 0.451885, 0.704048]
MixedL = [0.06427, 0.129408, 9.425478]

# Calculate in range of 0.5um to 1.8um
L = np.arange(0.5, 1.85, 0.00001)


# Sellmeier Equation to find n values
def SellmeierEquation(L, A, K):
    n2 = 1 + ((A[0] * (L * L)) / (L * L - K[0] * K[0])) + ((A[1] * (L * L)) / (L * L - K[1] * K[1])) + (
            (A[2] * (L * L)) / (L * L - K[2] * K[2]))
    n = np.sqrt(n2)
    return n


# Group Index Equation to find Ng values
def GroupIndex(n, L):
    Ng = [None]
    for i in range(0, len(n) - 1):
        Ng.append(n[i] - L[i] * ((n[i + 1] - n[i]) / (L[i + 1] - L[i])))
        i += 1
    return Ng


# Refractive Index for materials
PureSiO2 = SellmeierEquation(L, PureA, PureL)
MixedSiO2 = SellmeierEquation(L, MixedA, MixedL)

# Group Indices for materials
GI1 = GroupIndex(PureSiO2, L)  # PureSio2
GI2 = GroupIndex(MixedSiO2, L)  # MixedSi02

fig = mp.figure(figsize=(9, 6), tight_layout=True)

# Plot for Pure Medium
ax1 = fig.add_subplot(2, 2, 1)
Si02, = ax1.plot(L, PureSiO2, label="n", color='green')
Si02Ng, = ax1.plot(L, GI1, label="Ng", color='green', linestyle='dashed')
ax1.set(title="SiO", xlabel="Wavelength(\u03BCm)")
ax1.legend(handles=[Si02, Si02Ng], loc="best",ncol=2,frameon=False)

# Plot for Mixed Medium
ax2 = fig.add_subplot(2, 2, 2)
Si02GeO2, = ax2.plot(L, MixedSiO2, label="n", color='blue')
Si02GeO2Ng, = ax2.plot(L, GI2, label="Ng", color='blue', linestyle='dashed')
ax2.annotate('n', xytext=(1.3, 1.45), xy=(1.3, 1.45), c='blue',)
ax2.set(title="SiO2-GeO2", xlabel="Wavelength(\u03BCm)")
ax2.legend(handles=[Si02GeO2, Si02GeO2Ng], loc="best",ncol=2,frameon=False)


# Plot for Superposition of both mediums
ax3 = fig.add_subplot(2, 2, (3, 4))
ax3.plot(L, PureSiO2, color='green')
ax3.plot(L, GI1, color='green', linestyle='dashed')
ax3.plot(L, MixedSiO2, color='blue')
ax3.plot(L, GI2, color='blue', linestyle='dashed')
ax3.set(title="Group & Refractive Indices", xlabel="Wavelength(\u03BCm)")

mp.show()
