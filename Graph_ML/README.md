# SimulationsQMLSymmetries

# Hardware Experiments
- Used Python 3.10.11
- To install Requirements, upgrade pip to version 23.2.1 and run
```
pip install Graph_ML/requirements.txt
```
- Experiments were run for the Cn_circuit (cyclic invariant circuit) on the graph connectedness problem "nodes_8-graphs_3000.pt" with 8 quibits, 30 layers, and the first sample was used for the hardware / FakeTorino runs (relevant because of dataset shuffling). Seed used was always 42.
- 100 images were used for the training set, and 50 images were used for the test set (the first 50 of the remaining 2900 images)
- Experiments were run:
  - For 2 epochs on IBM Kingston hardware and FakeTorino Noisy Simulator, using the SPSA gradient
  - (As a comparsion, with both the SPSA and the default gradient on Pennylane)
  - For 50 epochs with the default gradient on Pennylane, before testing on IBM Kingston hardware
- Logs can be found in "Graph_ML/output"
- Output files (train & test labels, train & test predictions, weights and plots) can be found in "Graph_ML/output/connectedness". There, "read_me_experiments.txt" explains which id-numbers refer to which above-mentioned experiments.
- The following files were adapted / created for the experiments (and work in the current state only for this circuit & problem). Different configurations were made by changing / (un-) commenting code -> currently at last version (but commented real hardware).:
  - Graph_ML/script.sh to run (different configurations there from line 9)
  - Graph_ML/job.py and Graph_ML/just_test.py are mains for running / testing already trained model
  - Graph_ML/src/performance.py needed random seed for dataset split, smaller test set, sessions, etc.
  - The first few cells of Graph_ML/results_plotting_final.py and Graph_ML/plot.py for plotting / plotting after test of model trained on simulator
  - Graph_ML/credentials.py for credentials (not on github)
  - Graph_ML/requirements.txt

- Everything but credentials (and helper-files like Graph_ML/test_weights.py) can also be found on Github on branch "hardware_experiments_fixed": https://github.com/dmytro8765/Pennylane_ML/tree/hardware_experiments_fixed . Ask HiWi Dima Bondarenko for access if needed.