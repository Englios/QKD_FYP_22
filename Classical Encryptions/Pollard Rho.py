# Python 3 program to find a prime factor of composite using
# Pollard's Rho algorithm
import random
import math
import time
from matplotlib import pyplot as plt
from random import randint as rand
import numpy as np

t=[]
k=[]
integer=[]
intlen=100

# Function to calculate (base^exponent)%modulus
def modular_pow(base, exponent, modulus):
    # initialize result
    result = 1

    while (exponent > 0):

        # if y is odd, multiply base with result
        if (exponent & 1):
            result = (result * base) % modulus

        # exponent = exponent/2
        exponent = exponent >> 1

        # base = base * base
        base = (base * base) % modulus

    return result


# method to return prime divisor for n
def PollardRho(n):
    # no prime divisor for 1
    if (n == 1):
        return n

    # even number means one of the divisors is 2
    if (n % 2 == 0):
        return 2

    # we will pick from the range [2, N)
    x = (random.randint(0, 2) % (n - 2))
    y = x

    # the constant in f(x).
    # Algorithm can be re-run with a different c
    # if it throws failure for a composite.
    c = (random.randint(0, 1) % (n - 1))

    # Initialize candidate divisor (or result)
    d = 1

    # until the prime factor isn't obtained.
    # If n is prime, return n
    while (d == 1):

        # Tortoise Move: x(i+1) = f(x(i))
        x = (modular_pow(x, 2, n) + c + n) % n

        # Hare Move: y(i+1) = f(f(y(i)))
        y = (modular_pow(y, 2, n) + c + n) % n
        y = (modular_pow(y, 2, n) + c + n) % n

        # check gcd of |x-y| and n
        d = math.gcd(abs(x - y), n)

        # retry if the algorithm fails to find prime factor
        # with chosen x and c
        if (d == n):
            return PollardRho(n)

    return d


# Driver function
if __name__ == "__main__":
    try:
        j = 0
        for i in range(1, intlen + 1):
            print("--------------------------------------------------------------")
            # Get random Integer
            integer.append(rand(pow(10, i - 1), pow(10, i)))

            # Start clock time
            start = time.time()
            print('Iteration number', i)
            print('Number to be Factorized :', integer[i-1])
            print(PollardRho(integer[j]))

            # Get Final time to compute the script
            end = time.time() - start
            t.append(end)
            k.append(i)
            print(end, 'seconds')

            # plots graph
            plt.plot(k, t, 'o-',markersize=0.5)
            j += 1

        plt.title("Factorization of Integers Using Pollard Rho Method")
        plt.ylabel('time,s')
        plt.xlabel('no. of digits')
        plt.xticks(np.arange(min(k), max(k) + 1, 10))
        plt.show()

    except KeyboardInterrupt:
        plt.title("Factorization of Integers Using Pollard Rho Method")
        plt.ylabel('time,s')
        plt.xlabel('no. of digits')
        plt.xticks(np.arange(min(k), max(k) + 1, 10))
        plt.show()

# This code is contributed by chitranayal