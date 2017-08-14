# Genetic Algorithm for scikit-learn Neural Networks

This is an implementation of utilizing genetic algorithms for use in training neural networks by allowing the algorithm
to pick the architecture of the neural network. Neural networks are implemented using scikit-learn. Code was based off
Will Larson's article and modified to work with scikit-learn.

## Usage

Modify the main() of main.py to fit raining standards:

```
generations = 5  # Number of times to evolve the population.
population = 20  # Number of networks in each generation.
# Allows you to select from 0 to 1 how much you want to weight the speed of the neural network training vs the accuracy
time_weight = 0
# Network parameters, network_params[0]: tuple of layer ranges, network_params[1]: tuple of nodes per layer
network_params = [(0, 9), (1, 100)]
```

Calling generate() will commence the training, taking inputs X and outputs y
```
generate(generations, population, X, y, time_weight, network_params)
```

##### Example Output from main.py

```
***Evolving 3 generations with population 10***
***Doing generation 1 of 3***
    10 networks to train
    Training network: 0
    Training network: 1
    Training network: 2
    Training network: 3
    Training network: 4
    Training network: 5
    Training network: 6
    Training network: 7
    Training network: 8
    Training network: 9
Generation average loss: 0.85
Generation time average: 0.438288
--------------------------------------------------------------------------------
***Doing generation 2 of 3***
    10 networks to train
    Training network: 0
    Training network: 1
    Training network: 2
    Training network: 3
    Training network: 4
    Training network: 5
    Training network: 6
    Training network: 7
    Training network: 8
    Training network: 9
Generation average loss: 0.22
Generation time average: 0.556572
--------------------------------------------------------------------------------
***Doing generation 3 of 3***
    10 networks to train
    Training network: 0
    Training network: 1
    Training network: 2
    Training network: 3
    Training network: 4
    Training network: 5
    Training network: 6
    Training network: 7
    Training network: 8
    Training network: 9
Generation average loss: 0.27
Generation time average: 0.826589
--------------------------------------------------------------------------------
Top 5 networks:
Network #0 Loss: 0.204850427143
    Hidden Layer: 0 Nodes: 30
    Hidden Layer: 1 Nodes: 86
    Hidden Layer: 2 Nodes: 86
    Hidden Layer: 3 Nodes: 50
    Hidden Layer: 4 Nodes: 50
    Hidden Layer: 5 Nodes: 86
    Hidden Layer: 6 Nodes: 30
    Hidden Layer: 7 Nodes: 30
    Hidden Layer: 8 Nodes: 86
    Hidden Layer: 9 Nodes: 30
  Time: 0.6316337585449219
Network #1 Loss: 0.217460656249
    Hidden Layer: 0 Nodes: 30
    Hidden Layer: 1 Nodes: 30
    Hidden Layer: 2 Nodes: 86
    Hidden Layer: 3 Nodes: 30
    Hidden Layer: 4 Nodes: 86
    Hidden Layer: 5 Nodes: 30
    Hidden Layer: 6 Nodes: 30
    Hidden Layer: 7 Nodes: 30
    Hidden Layer: 8 Nodes: 30
    Hidden Layer: 9 Nodes: 30
  Time: 0.899622917175293
Network #2 Loss: 0.224268311332
    Hidden Layer: 0 Nodes: 30
    Hidden Layer: 1 Nodes: 30
    Hidden Layer: 2 Nodes: 30
    Hidden Layer: 3 Nodes: 86
    Hidden Layer: 4 Nodes: 30
    Hidden Layer: 5 Nodes: 86
    Hidden Layer: 6 Nodes: 30
    Hidden Layer: 7 Nodes: 50
    Hidden Layer: 8 Nodes: 30
    Hidden Layer: 9 Nodes: 30
  Time: 0.5387742519378662
Network #3 Loss: 0.22683541315
    Hidden Layer: 0 Nodes: 30
    Hidden Layer: 1 Nodes: 68
    Hidden Layer: 2 Nodes: 50
    Hidden Layer: 3 Nodes: 66
    Hidden Layer: 4 Nodes: 77
    Hidden Layer: 5 Nodes: 38
    Hidden Layer: 6 Nodes: 57
  Time: 0.5893151760101318
Network #4 Loss: 0.229264038112
    Hidden Layer: 0 Nodes: 30
    Hidden Layer: 1 Nodes: 50
    Hidden Layer: 2 Nodes: 57
    Hidden Layer: 3 Nodes: 57
    Hidden Layer: 4 Nodes: 57
    Hidden Layer: 5 Nodes: 57
    Hidden Layer: 6 Nodes: 30
    Hidden Layer: 7 Nodes: 30
    Hidden Layer: 8 Nodes: 50
    Hidden Layer: 9 Nodes: 30
  Time: 1.0743348598480225
```

## Built With

* [NumPy](http://www.numpy.org/) - The matrix framework used
* [scikit-learn](http://scikit-learn.org/stable/) - framework testing against

## Inspired by
* [Will Larson](https://lethain.com/genetic-algorithms-cool-name-damn-simple/)
