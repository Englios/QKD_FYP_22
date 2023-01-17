# https://github.com/awnonbhowmik/RSA-Python/blob/master/RSA_Python.py

import math
# sqrt()fn

import random
from random import randint as rand

# rand range and other functions

maxprimelenght = 1000


# Extended Euclids Algorithm for Modular Inverse calculation
def egcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        g, x, y = egcd(b % a, a)

    return g, y - b // a * x, x


# Euclid's Algorithm for greatest common divisor
def gcd(a, b):
    while b != 0:
        # Swap Variables using tuple
        a, b = b, a % b
    return a


# Modular Inverse of the function
def mod_inverse(a, m):
    g, x, _ = egcd(a, m)
    if g != 1:
        raise Exception('gcd(a,m) != 1')
    return x % m


# Checking if number is prime
def isprime(n):
    if n < 2:
        return False
    elif n == 2:
        return True
    else:
        for i in range(2, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
    return True


# Generate random prime numbers
def generateRandomPrim():
    while 1:
        ranPrime = rand(1, maxprimelenght)
        if isprime(ranPrime):
            return ranPrime


# Generate Key pairs
def generateKeypairs():
    p = generateRandomPrim()
    q = generateRandomPrim()

    while p == q:
        q = generateRandomPrim()

    n = p * q
    phi = (p - 1) * (q - 1)
    e = rand(1, phi)

    while e < phi:
        if gcd(e, phi) == 1:
            break
        else:
            e = rand(1, phi)

    d = round(mod_inverse(e, phi))

    # print(e * d % phi)

    if e < phi:
        # print(p, q, n, phi, e, d)
        print('', 'Public Key: ', (e, n),
              "\n",
              'Public Key: ', (d, n))

    return (e, n), (d, n)


def encrypt(ptext, pubkey):
    key, n = pubkey
    cipher = [pow(ord(char), key, n) for char in ptext]

    return cipher


def decrypt(ctext, private_key):
    try:
        key, n = private_key
        text = [chr(pow(char, key, n)) for char in ctext]
        return "".join(text)
    except TypeError as e:
        print(e)


if __name__ == '__main__':
    public_key, private_key = generateKeypairs()
    print("Public: ", public_key)
    print("Private: ", private_key)

    ctext = encrypt(input("Get Messsage :"), public_key)
    print("encrypted  =", ctext)
    plaintext = decrypt(ctext, private_key)
    print("decrypted =", plaintext)
