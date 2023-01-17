#https://www.pythonpool.com/rsa-encryption-python/#:~:text=Report%20Ad-,What%20is%20RSA%20Encryption%20in%20python%3F,key%20and%20the%20private%20key.
import math
import numpy as np

seed = int(input("Enter Seed :"))
message= int(input("Enter messsage to be encrypted: "))

p=np.random.randint(seed)
q=np.random.randint(seed)
e=p-q

n=p*q

print("P value is",p)
print("Q value is",q)

def encrpyt(me):
    en = math.pow(me,e)
    c=en%n
    print("Encrypted message is :",c)
    return c

print("Original Message is ",message)
c=encrpyt(message)