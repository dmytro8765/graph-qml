from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.random import random_circuit
from credentials import TOKEN, CRN

service = QiskitRuntimeService.save_account(token=TOKEN, instance=CRN, set_as_default=True, overwrite=True)

service = QiskitRuntimeService()

print(service.backends())
backend = service.backend("ibm_kingston")
print(f"Backend: {backend}")

properties = backend.properties()

for gate in properties.gates:
    name = gate.gate
    print(f"Gate: {name}")
    for param in gate.parameters:
        print(f"param: {param}")

qc = random_circuit(num_qubits=8, depth=5)
qc.measure_all()
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
transpiled_qc = pm.run(qc)
print(f"Transpiled depth: {transpiled_qc.depth()}")

sampler = Sampler(backend)
sampler.options.default_shots = 100

job = sampler.run([transpiled_qc])
print(f"Job id: {job.job_id()}\nJob Status: {job.status()}")
result = job.result()
for pub_result in result:
    print(pub_result.data.meas.get_counts())
