start_time=$(date +%s)

##python3 job.py -l 3 -q 6 -n 1 -c "Energy_circuit" -b "/Users/home/qiskit_env/Pennylane/data/clique_supervised" -d "nodes_6-graphs_3000_per_qubit.pt" -e 150 -s 10
#python3 job.py -l 20 -q 6 -n 1 -c "Sn_circuit" -t "Max_Cut" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/max_cut" -d "nodes_6-graphs_3000_per_qubit.pt" -e 5 -s 5
#python3 job.py -l 20 -q 6 -n 0 -c "Cn_circuit" -t "Max_Cut" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/max_cut" -d "nodes_6-graphs_3000_per_qubit.pt" -e 5 -s 5
#python3 job.py -l 10 -q 6 -n 0 -c "entanglement_circuit" -t "Max_Cut" -b "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/max_cut" -d "nodes_6-graphs_3000_per_qubit.pt" -e 5 -s 5
#python3 job.py -l 2 -q 8 -n 0 -c "Sn_free_parameter_circuit" -b "/Users/home/qiskit_env/Pennylane/data/graph_connectedness" -d "nodes_8-graphs_3000-edges_8_to_17.pt" -e 100 -s 5

#python3 job.py -l 30 -q 8 -n 27061720 -c "Cn_circuit" -t "Connectedness" -b "/Users/danielles/Documents/Projekte/BAIQO/invariant_circuits/Pennylane_ML" -d "nodes_8-graphs_3000.pt" -e 100 -s 10
#python3.10 job.py -l 30 -q 8 -n 29060424 -c "Cn_circuit" -t "Connectedness" -b "/Users/danielle/BAIQO/Pennylane_ML" -d "nodes_8-graphs_3000.pt" -e 10 -s 1 -r 0 2>&1 | tee /Users/danielle/BAIQO/Pennylane_ML/Graph_ML/output/console_log_ex_3.txt
#python3.10 job.py -l 30 -q 8 -n 29060424 -c "Cn_circuit" -t "Connectedness" -b "/home/schuman/Pennylane_ML" -d "nodes_8-graphs_3000.pt" -e 10 -s 1 -r 0 -br 2 2>&1 | tee /home/schuman/Pennylane_ML/Graph_ML/output/console_log_ex_4.txt
#python3.10 job.py -l 30 -q 8 -n 07072256 -c "Cn_circuit" -t "Connectedness" -b "/home/schuman/Pennylane_ML" -d "nodes_8-graphs_3000.pt" -e 2 -s 1 -r 0 -br 2 2>&1 | tee /home/schuman/Pennylane_ML/Graph_ML/output/console_log_test.txt
# python3.10 job.py -l 30 -q 8 -n 07081426 -c "Cn_circuit" -t "Connectedness" -b "/home/schuman/Pennylane_ML" -d "nodes_8-graphs_3000.pt" -e 15 -s 1 2>&1 | tee /home/schuman/Pennylane_ML/Graph_ML/output/console_log_ex_fake_torino.txt
python3.10 job.py -l 30 -q 8 -n 07251122 -c "Cn_circuit" -t "Connectedness" -b "/home/schuman/Pennylane_ML" -d "nodes_8-graphs_3000.pt" -e 50 -s 10 2>&1 | tee /home/schuman/Pennylane_ML/Graph_ML/output/pennylane_reproduction.txt

end_time=$(date +%s)  # Capture end time
elapsed_time=$((end_time - start_time))  # Compute duration

echo "Total execution time: $elapsed_time seconds"