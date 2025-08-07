"""
This script contains commands used to launch experiments 
for various quantum circuit architectures on subgraph isomorphism tasks. 

Each command sets specific hyperparameters to achieve a target number of trainable parameters 
(e.g., 15, 30, 60, 90, 120), enabling systematic benchmarking. 

The final command summarizes the results across all circuits for analysis.
"""

# Parameter count targets: 15, 30, 60, 90, 120.
# 15 -> layers: 4, 4, 4, 2, 2, 2, 5, 5, 5
# 30 -> layers: 7, 7, 7, 5, 5, 5, 10, 10, 10
# 60 -> layers: 15, 15, 15, 10, 10, 10, 20, 20, 20
# 90 -> layers: 22, 22, 22, 15, 15, 15, 30, 30, 30
# 120 -> layers: 30, 30, 30, 20, 20, 20, 40, 40, 40

# Parameters: 4 x layers
#python3 job_subgraph.py -l 22 -q 6 -sub 4 -n 0 -c "1" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_per_qubit_3000_isomorph.pt" -e 50 -s 3

# Parameters: 4 x layers
python3 job_subgraph.py -l 22 -q 6 -sub 4 -n 0 -c "2" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_per_qubit_3000_isomorph.pt" -e 50 -s 3

# Parameters: 4 x layers + 1
#python3 job_subgraph.py -l 22 -q 6 -sub 4 -n 0 -c "3" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_per_qubit_3000_isomorph.pt" -e 50 -s 3

# Parameters: 6 x layers
#python3 job_subgraph.py -l 15 -q 6 -sub 4 -n 0 -c "4" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_per_qubit_3000_isomorph.pt" -e 50 -s 3

# Parameters: 6 x layers
#python3 job_subgraph.py -l 15 -q 6 -sub 4 -n 0 -c "5" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_per_qubit_3000_isomorph.pt" -e 50 -s 3

# Parameters: 6 x layers + 1
#python3 job_subgraph.py -l 15 -q 6 -sub 4 -n 0 -c "6" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_per_qubit_3000_isomorph.pt" -e 50 -s 3

# Parameters: 3 x layers
# python3 job_subgraph.py -l 30 -q 6 -sub 4 -n 0 -c "7" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_per_qubit_3000_isomorph.pt" -e 50 -s 3

# Parameters: 3 x layers
#python3 job_subgraph.py -l 30 -q 6 -sub 4 -n 0 -c "8" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_per_qubit_3000_isomorph.pt" -e 50 -s 3

# Parameters: 3 x layers + 1
#python3 job_subgraph.py -l 30 -q 6 -sub 4 -n 0 -c "9" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_per_qubit_3000_isomorph.pt" -e 50 -s 3

# Process the results:
# python3 log_summary_wandb.py -q 6 -c 1 2 3 4 5 6 7 8 9 -n 0 -f "y" -e 50 -s 3 -i 2900 -pt 90