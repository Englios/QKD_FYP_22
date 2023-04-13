from abc import ABCMeta,abstractmethod
from contextlib import contextmanager

class Qubit(metaclass=ABCMeta):
    
    @abstractmethod
    def h(self): pass
    
    @abstractmethod
    def measure(self)-> bool:pass
    
    @abstractmethod
    def reset(self): pass

class QuantumDevice (metaclass=ABCMeta):
    
    @abstractmethod
    def allocate_qubit(self):
        pass
    
    @abstractmethod
    def deallocate_qubits(self):
        pass
    
    @abstractmethod
    def using_qubit(self):
        qubit=self.allocate_qubit()
        try:
            yield qubit
        finally:
            qubit.reset()
            self.deallocate_qubits(qubit)