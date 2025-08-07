start_time=$(date +%s)

python3 job.py -l 4 -q 6 -sub 4 -n 0 -c "1" -t "Subgraph" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_3000.pt" -e 10 -s 3
python3 job.py -l 4 -q 6 -sub 4 -n 0 -c "2" -t "Subgraph" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_3000.pt" -e 10 -s 3
#python3 job.py -l 10 -q 6 -sub 4 -n 0 -c "3" -t "Subgraph" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_3000.pt" -e 10 -s 1
#python3 job.py -l 10 -q 6 -sub 4 -n 0 -c "4" -t "Subgraph" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_3000.pt" -e 10 -s 1
#python3 job.py -l 10 -q 6 -sub 4 -n 0 -c "5" -t "Subgraph" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_3000.pt" -e 10 -s 1
#python3 job.py -l 10 -q 6 -sub 4 -n 0 -c "6" -t "Subgraph" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_3000.pt" -e 10 -s 1
#python3 job.py -l 10 -q 6 -sub 4 -n 0 -c "7" -t "Subgraph" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_3000.pt" -e 10 -s 1
#python3 job.py -l 10 -q 6 -sub 4 -n 0 -c "8" -t "Subgraph" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_3000.pt" -e 10 -s 1
#python3 job.py -l 5 -q 6 -sub 4 -n 0 -c "9" -t "Subgraph" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_3000.pt" -e 10 -s 1

end_time=$(date +%s)  # Capture end time
elapsed_time=$((end_time - start_time))  # Compute duration

echo "Total execution time: $elapsed_time seconds"

#python3 job.py -l 3 -q 6 -n 1 -c "Energy_circuit" -b "/Users/home/qiskit_env/Pennylane/data/clique_supervised" -d "nodes_6-graphs_3000_per_qubit.pt" -e 150 -s 10
#python3 job.py -l 1 -q 6 -n 1 -c "Sn_circuit" -t "Clique" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/clique_supervised" -d "nodes_6-graphs_3000_per_qubit.pt" -e 5 -s 5
#python3 job.py -l 20 -q 6 -n 0 -c "Cn_circuit" -t "Connectedness" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/graph_connectedness" -d "nodes_6-graphs_3000-edges_5_6_7.pt" -e 5 -s 5
#python3 job.py -l 10 -q 6 -n 0 -c "entanglement_circuit" -t "Connectedness" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/graph_connectedness" -d "nodes_6-graphs_3000-edges_5_6_7.pt" -e 5 -s 5
#python3 job.py -l 2 -q 8 -n 0 -c "Sn_free_parameter_circuit" -t "Connectedness" -b "/Users/home/qiskit_env/Pennylane/data/graph_connectedness" -d "nodes_8-graphs_3000-edges_8_to_17.pt" -e 100 -s 5
#python3 job.py -l 2 -q 6 -sub 4 -n 0 -c "Sub_Sn_circuit_app2" -t "Subgraph" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_3000.pt" -e 10 -s 1
#python3 job.py -l 2 -q 6 -sub 4 -n 0 -c "Sub_Sn_circuit_app3" -t "Subgraph" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph" -d "graph-6_subgraph-4_3000.pt" -e 10 -s 1
