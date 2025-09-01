#!/usr/bin/env bash
#
#SBATCH --job-name="Graph Connectedness"
#SBATCH --chdir=/home/w/woelckert/work/symmetrie-learning/SimulationsQMLSymmetries
#SBATCH --partition=All

# Setup pyenv shell
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate venv 1>/dev/null 2>&1

# QUICK GUIDE PARAMETERS
# l: number of layers (see `weight_shapes` in 'job.py' to calculate actual number of parameters)
# q: number of qubits (has to align with used dataset)
# n: special number to append to output filenames if one wants to prevent overwrites
# b: base directory where this file is located (use absolute paths)
# d: name of the dataset located in 'b/data/graph_connectedness/'
# e: number of epochs for training
# s: number of subsamplings (number of training/test samples)


# dont forget to also change the slurm directory
base_directory="/home/w/woelckert/work/symmetrie-learning/SimulationsQMLSymmetries"
dataset_6_qubits="nodes_6-graphs_3000-edges_5_6_7.pt"
dataset_8_qubits="nodes_8-graphs_3000-edges_8_to_17.pt"
dataset_10_qubits="nodes_10-graphs_3000-edges_9_to_28.pt"


# STANDARD CIRCUITS
# SETUP FOR 6 QUBITS AND 120 PARAMETERS
# python3 job.py -l 40 -q 6 -n 0 -c "Sn_circuit" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30
# python3 job.py -l 20 -q 6 -n 0 -c "entanglement_circuit" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30
# python3 job.py -l 7 -q 6 -n 0 -c "strongly_entanglement_circuit" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30
# python3 job.py -l 40 -q 6 -n 0 -c "Cn_circuit" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30
# python3 job.py -l 30 -q 6 -n 0 -c "Cn_circuit2" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30

# SETUP FOR 8 QUBITS AND 120 PARAMETERS
# python3 job.py -l 40 -q 8 -n 0 -c "Sn_circuit" -b "$base_directory" -d "$dataset_8_qubits" -e 50 -s 30
# python3 job.py -l 15 -q 8 -n 0 -c "entanglement_circuit" -b "$base_directory" -d "$dataset_8_qubits" -e 50 -s 30
# python3 job.py -l 5 -q 8 -n 0 -c "strongly_entanglement_circuit" -b "$base_directory" -d "$dataset_8_qubits" -e 50 -s 30
# python3 job.py -l 40 -q 8 -n 0 -c "Cn_circuit" -b "$base_directory" -d "$dataset_8_qubits" -e 50 -s 30
# python3 job.py -l 30 -q 8 -n 0 -c "Cn_circuit2" -b "$base_directory" -d "$dataset_8_qubits" -e 50 -s 30

# SETUP FOR 8 QUBITS AND 240 PARAMETERS
# python3 job.py -l 80 -q 8 -n 0 -c "Sn_circuit" -b "$base_directory" -d "$dataset_8_qubits" -e 50 -s 30
# python3 job.py -l 30 -q 8 -n 0 -c "entanglement_circuit" -b "$base_directory" -d "$dataset_8_qubits" -e 50 -s 30
# python3 job.py -l 10 -q 8 -n 0 -c "strongly_entanglement_circuit" -b "$base_directory" -d "$dataset_8_qubits" -e 50 -s 30
# python3 job.py -l 80 -q 8 -n 0 -c "Cn_circuit" -b "$base_directory" -d "$dataset_8_qubits" -e 50 -s 30
# python3 job.py -l 60 -q 8 -n 0 -c "Cn_circuit2" -b "$base_directory" -d "$dataset_8_qubits" -e 50 -s 30


# NON SYMMETRIC CIRCUITS
# SETUP FOR 6 QUBITS AND 120 PARAMETERS
# python3 job.py -l 40 -q 6 -n 0 -c "Cn_circuit_with_anti_part" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30
# python3 job.py -l 38 -rl 1 -q 6 -n 0 -c "Sn_circuit_with_individual_x_rotations" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30
# python3 job.py -l 20 -rl 8 -q 6 -n 1 -c "Sn_circuit_with_individual_x_rotations" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30
# python3 job.py -l 30 -rl 4 -q 6 -n 2 -c "Sn_circuit_with_individual_x_rotations" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30
# python3 job.py -l 1 -rl 15 -q 6 -n 3 -c "Sn_circuit_with_individual_x_rotations" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30
# python3 job.py -l 0 -rl 16 -q 6 -n 4 -c "Sn_circuit_with_individual_x_rotations" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30
# python3 job.py -l 9 -rl 12 -q 6 -n 5 -c "Sn_circuit_with_individual_x_rotations" -b "$base_directory" -d "$dataset_6_qubits" -e 50 -s 30