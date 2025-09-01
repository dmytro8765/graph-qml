# This script contains commands used to launch experiments 
# for various quantum circuit architectures on subgraph isomorphism tasks. 

# Each command sets specific hyperparameters to achieve a target number of trainable parameters 
# (e.g., 15, 30, 60, 90, 120), enabling systematic benchmarking. 

# The final command summarizes the results across all circuits for analysis.

# For baseline circuit "0": 30 parameter per layer!

# Parameter count targets: 15, 30, 60, 90, 120.
# 15 ->  layers: 4,   4,   4,   2,   2,   2,   5,   5,   5
# 30 ->  layers: 7,   7,   7,   5,   5,   5,   10,  10,  10
# 60 ->  layers: 15,  15,  15,  10,  10,  10,  20,  20,  20
# 90 ->  layers: 22,  22,  22,  15,  15,  15,  30,  30,  30
# 120 -> layers: 30,  30,  30,  20,  20,  20,  40,  40,  40

#!/bin/bash
cd "$(dirname "$0")/.." # always move to the root directory (here: Pennylane)

# Parameters: 1 x layers
# python3 -m Subgraph.job_subgraph -l 3 -q 6 -sub 4 -n 0 -c "0" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Subgraph/node/data/datasets" -dtrain "graph-6_subgraph-4_3000_iso_train.pt" -dtest "graph-6_subgraph-4_3000_iso_test.pt" -e 10 -s 2

# Parameters: 4 x layers
# python3 -m Subgraph.job_subgraph -l 4 -q 6 -sub 4 -n 0 -c "1" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Subgraph/node/data/datasets" -dtrain "graph-6_subgraph-4_3000_iso_train.pt" -dtest "graph-6_subgraph-4_3000_iso_test.pt" -e 20 -s 2

# Parameters: 4 x layers
#python3 -m Subgraph.job_subgraph -l 22 -q 6 -sub 4 -n 0 -c "2" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Subgraph/node/data/datasets" -dtrain "graph-6_subgraph-4_3000_iso_train.pt" -dtest "graph-6_subgraph-4_3000_iso_test.pt" -e 50 -s 3

# Parameters: 4 x layers + 1
#python3 -m Subgraph.job_subgraph -l 22 -q 6 -sub 4 -n 0 -c "3" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Subgraph/node/data/datasets" -dtrain "graph-6_subgraph-4_3000_iso_train.pt" -dtest "graph-6_subgraph-4_3000_iso_test.pt" -e 50 -s 3

# Parameters: 6 x layers
#python3 -m Subgraph.job_subgraph -l 15 -q 6 -sub 4 -n 0 -c "4" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Subgraph/node/data/datasets" -dtrain "graph-6_subgraph-4_3000_iso_train.pt" -dtest "graph-6_subgraph-4_3000_iso_test.pt" -e 50 -s 3

# Parameters: 6 x layers
#python3 -m Subgraph.job_subgraph -l 15 -q 6 -sub 4 -n 0 -c "5" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Subgraph/node/data/datasets" -dtrain "graph-6_subgraph-4_3000_iso_train.pt" -dtest "graph-6_subgraph-4_3000_iso_test.pt" -e 50 -s 3

# Parameters: 6 x layers + 1
python3 -m Subgraph.job_subgraph -l 10 -q 6 -sub 4 -n 0 -c "6" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Subgraph/node/data/datasets" -dtrain "graph-6_subgraph-4_3000_iso_train.pt" -dtest "graph-6_subgraph-4_3000_iso_test.pt" -e 10 -s 2

# Parameters: 3 x layers
# python3 -m Subgraph.job_subgraph -l 10 -q 6 -sub 4 -n 0 -c "7" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Subgraph/node/data/datasets" -dtrain "graph-6_subgraph-4_3000_iso_train.pt" -dtest "graph-6_subgraph-4_3000_iso_test.pt" -e 5 -s 2

# Parameters: 3 x layers
#python3 -m Subgraph.job_subgraph -l 30 -q 6 -sub 4 -n 0 -c "8" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Subgraph/node/data/datasets" -dtrain "graph-6_subgraph-4_3000_iso_train.pt" -dtest "graph-6_subgraph-4_3000_iso_test.pt" -e 25 -s 2

# Parameters: 3 x layers + 1
#python3 -m Subgraph.job_subgraph -l 30 -q 6 -sub 4 -n 0 -c "9" -f "y" -b "/Users/home/Quantum_Computing/Pennylane/Subgraph/node/data/datasets" -dtrain "graph-6_subgraph-4_3000_iso_train.pt" -dtest "graph-6_subgraph-4_3000_iso_test.pt" -e 25 -s 2

# Process the results:
python3 -m Subgraph.log_summary_wandb -q 6 -sub 4 -c 6 -n 0 -f "y" -e 10 -s 2 -i 2900 -pt 60