import random
import math

def gcd(a,b):
    if b==0:
        return a
    else:
        return gcd(b,a%b)
    
def multiplicative_inverse(e,phi):
    for d in range(1,phi):
        if (d*e) % phi ==1:
            return d
    return None

def is_prime(n):
    if n<2:
        return False
    for i in range(2,int(math.sqrt(n)+1)):
        if n % i == 0:
            return False
    return True

def generate_keypair(p,q):
    if not (is_prime(p) and is_prime(q)):
        raise ValueError("Both Numbers must be prime")
        
    elif p==q:
        raise ValueError("p and q cannot be equal")
    
    n=p*q
    
    phi =(p-1)*(q-1)
    
    e=random.randrange(1,phi)
    g=gcd(e,phi)
    
    while g != 1:
        e=random.randrange(1,phi)
        g=gcd(e,phi)
        
    d= multiplicative_inverse(e,phi)
    
    return ((n,e),(n,d))

def encrypt(public_key,plaintext):
    n,e=public_key
    
    cipher=[pow(ord(char),e,n) for char in plaintext]
    
    return cipher

def decrypt(private_key,ciphertext):
    n,d=private_key
    
    plain=[chr(pow(char,d,n)) for char in ciphertext]
    
    return ''.join(plain)

def main():
    p=random.randint(1,100)
    q=random.randint(1,100)
    
    while generate_keypair(p,q) == ValueError:
        p=random.randint(1,100)
        q=random.randint(1,100)
        
    public,private=generate_keypair(p,q)
    
    print("Public key: ", public)
    print("Private key: ", private)
    message = str(input('Enter Message: '))
    encrypted_message = encrypt(public, message)
    print("Encrypted message: ", encrypted_message)
    decrypted_message = decrypt(private, encrypted_message)
    print("Decrypted message: ", decrypted_message)
    
if __name__=='__main__':
    main()