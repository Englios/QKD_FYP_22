import random
import time
import pandas as pd
import numpy as np
from random import randint as rand
from matplotlib import pyplot as plt

#Arrays
t = [] #Store Time
k = [] #Store Lenght of Integer
integer=[] #Store Integers
intlen = 30


# Get Prime Factors of the number
def get_primefactors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


try:
    j = 0
    for i in range(1, intlen + 1):
        print("--------------------------------------------------------------")
        # Get random Integer
        integer.append(rand(pow(10, i - 1), pow(10, i))) #ensure lenght of integer increase linearly

        # Start clock time
        start = time.time()
        print('Iteration number', i)
        print('Number to be Factorized :', integer[j])

        #Factorization using get_primefactor function
        print(get_primefactors(integer[j]))

        # Get Final time to compute the script
        end = time.time() - start
        t.append(end)
        k.append(i)
        print(end, 'seconds')

        # plots graph
        plt.plot(k, t, '-o')
        j += 1

    plt.title("Factorization of Integers Using Brute Force Method")
    plt.ylabel('time,s')
    plt.xlabel('no. of digits')
    plt.xticks(np.arange(min(k), max(k) + 1, 1))
    plt.show()

except KeyboardInterrupt: #manually interrupt code incase of long compute times
    plt.title("Factorization of Integers Using Brute Force Method")
    plt.ylabel('time,s')
    plt.xlabel('no. of digits')
    plt.xticks(np.arange(min(k), max(k) + 1, 1))
    plt.show()

#making pandas dataframe
data = {
    'Integer':integer,
    'time': t,
    'no of digits': k
}

#Save data to excel sheet
df1 = pd.DataFrame(data)
writer = pd.ExcelWriter(r'C:\Users\alifa\OneDrive\Documents\Universitas Malaya\FYP\Quantum\Seeded CSC\Unseeded Data.xlsx',mode='a')
df1.to_excel(writer, index=False, sheet_name='5')
writer.save()
