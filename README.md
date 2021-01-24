# Model-Agnostic Meta-Learning Supervised Regression Experiments

This repository contains code that can be used to run supervised regression experiments with model-agnostic meta-learning (MAML) neural networks and sine wave functions.

## Installation

It is recommended to install conda and then create an environment for the software using the ```environment.yaml``` file. A suggestion on how to install the software and activate the environment is provided below.

```bash
git clone https://github.com/anpenta/model-agnostic-meta-learning-supervised-regression-experiments.git
cd model-agnostic-meta-learning-supervised-regression-experiments
conda env create -f environment.yaml
conda activate model-agnostic-meta-learning-supervised-regression-experiments
```

## Running the experiments

To run the experiments you can provide commands through the terminal using the ```run_experiment``` module. An example is given below.

```bash
python3 code/run_experiment.py 10000 1000 10 5 10 ./experimental-results
```
This will run a supervised regression experiment with 10000 baseline training iterations, 1000 meta-training iterations, a meta-batch size of 10, a batch size of 5, and a random seed of 10, and save the experimental results in the ./experimental-results directory. An example of how to see the arguments for running the experiments is provided below.

```bash
python3 run_experiment.py --help
```

## Results

As an example, below are some experimental results.

<p float="left">
<img src=./experimental-results/baseline-test-sine-wave-approximation.png height="320" width="420">
<img src=./experimental-results/baseline-test-training-loss.png height="320" width="420">
</p>

<p float="left">
<img src=./experimental-results/maml-test-sine-wave-approximation.png height="320" width="420">
<img src=./experimental-results/maml-test-training-loss.png height="320" width="420">
</p>

## Sources
* Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks." arXiv preprint arXiv:1703.03400 (2017).
