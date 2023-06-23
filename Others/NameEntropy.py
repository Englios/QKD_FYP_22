import numpy as np

name = 'muhammad alif aiman bin abdul malek'

length = len(name)  # length of name
print(length)

#remove space
name = name.replace(' ', '')
print(len(name))

#find unique character
unique_char = np.unique(list(name))
print(unique_char)

#find frequency of each character
frequency = [] 
for i in unique_char:
    frequency.append(name.count(i))
print(frequency)

