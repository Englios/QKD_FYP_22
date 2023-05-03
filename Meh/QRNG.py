from qiskit import *
from matplotlib import pyplot as mpl
from qiskit.tools.visualization import plot_histogram,plot_bloch_vector
# import matplotlib_inline

qr=QuantumRegister(5)
cr=ClassicalRegister(5)
circuit=QuantumCircuit(qr,cr)

circuit.h(qr)
circuit.measure(qr,cr)
circuit.draw(output='mpl',scale=1,filename='Test')
# mpl.show()

backend=Aer.get_backend('qasm_simulator')
result = execute(circuit, backend, shots=100, memory=True).result()

rawvalues_sim = result.get_memory()
print(rawvalues_sim)