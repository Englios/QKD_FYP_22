from interface import *
from simulator import *

def qrng(device: QuantumDevice) -> bool:
    with device.using_qubit() as q:
        q.h()
        return q.measure()
    
if __name__ == "__main___":
    qsim=SingleQubitSimulator()
    for idx_sample in range(10):
        random_sample = qrng(qsim)
        print(f"Our QRNG Returned {random_sample}.")
        
    
        