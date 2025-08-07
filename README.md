# Solving graph problems with Quantum Machine Learning
Welcome!

In this repository, we conduct a training process of neural networks that have various quantum circuits as their basis, in attempt to solve famous graph problems. 

## Detailed manual 📖
In the works... ⌛
In the meantime, for subgraph-related files (and some others): see the comment at the top of each said file for brief description of the contents.

## Enviroment ⚙️
- Python 3.12.2 🐍
- Install requirements:
```setup
pip install -r requirements.txt
```
This can be done either directly, or in a virtual enviroment, which can be created with:
```setup
python3 -m venv venv
```
and activated with:
```setup
source venv/bin/activate
```
## Framework 🛠️
This projects consist of 3 main training parts:
1) Generating datasets, consisting of graphs and labels, for each graph problem at hand: **data**. 🗄️
2) Training the neural network with a quantum circuit as its basic trainable unit: **src** and **job.py**, **job_subgraph.py**. 🤖
3) Computing the accuracy of the model by comparing the outputs with true labels: **results_plotting_final** and **log_summary_wand.py**. 📈

### Datasets 🗄️
The folder **data** consists of multiple folders, such as **bipartite**, **hamiltonian** etc. (each folder name is given by the graph problem it corresponds to). **subgraph**, an example of the latest graph problem at hand, consists of multiple **.py** files, which generate a pytorch dataset, consisting of a main graph, subgraph to be found, and label with all possible locations of the subgraph inside a big graph.

### Training 🤖
Called **job.py** or **job_subgraph.py**, these files 
- take multiple parameters, such as number of layers, samplings, epochs;
- split the input dataset in a suitable manner; 
- perform a neural network training, contained in files suh as **performance.py**, **performance_subgraph.py** etc, in the **src** folder.

### Evaluation 📈
Finally, the result of the neural network in the last epoch, obtained in the previous step, is compared to the true target values, stored in the dataset, and the final accuracy is computed and plotted.
Files:
1) **results_plotting_final.py**: for all tasks outside of the subgraph problem.
2) **log_summary_wandb.py**: for the subgraph problem, includes logging into a Weights and Biases Project.

## Documentation 📝
Additionally, the training process, along with final accuracies and plots, are logged to a Weights and Biases Project web-page, where, when viewing by **Run**, each training process can be observed and found in detail. This feature can be deactivated, otherwise, another link to an appropriate WandB server has to be established before executing the training process.

## Execution ✅
To execute the entire process of training and to compute and plot the accuracy development, a shell script is used. In the file **script.sh** or **script_subgraph.sh**, one has to adjust parameters, mentioned above (layer count, circuit type, dataset name etc) and run 
```setup
python3 sh script.sh
```
