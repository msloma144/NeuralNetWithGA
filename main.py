import GA
from sklearn.datasets import load_breast_cancer
import time

def print_top_networks(networks):
    print("Top " + str(len(networks)) + " networks:")
    for i in range(0, len(networks)):
        print("Network #" + str(i) + " Loss: " + str(networks[i][0].__getattribute__('loss_')))
        for j in range(0, len(networks[i][0].__getattribute__('coefs_'))):
            print("    Hidden Layer: " + str(j) + " Nodes: " + str(len(networks[i][0].__getattribute__('coefs_')[j])))
        print("  Time: " + str(networks[i][1]))


def train_networks(networks, X, y):
    # Train the neural networks
    i = 0
    print("    " + str(len(networks)) + " networks to train")
    for network in networks:
        start = time.time()
        network[0].fit(X, y)
        end = time.time()
        print("    Training network: " + str(i))
        total_time = end - start
        network[1] = total_time
        i += 1

def get_average_accuracy(networks):
    # Get the average accuracy for a group of networks
    total_accuracy = 0
    total_time = 0
    for network in networks:
        total_accuracy += network[0].__getattribute__('loss_')
        total_time += network[1]

    return [(total_accuracy / len(networks)), (total_time / len(networks))]


def generate(generations, population, X, y, time_weight, network_params):
    # Generate a network with the genetic algorithm
    optimizer = GA.Optimizer()
    networks = optimizer.create_population(population, network_params)

    # Evolve the generation.
    for i in range(generations):
        print("***Doing generation %d of %d***" %
              (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, X, y)

        # Get the average accuracy for this generation.
        average_accuracy_n_time = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        print("Generation average loss: %.2f" % (average_accuracy_n_time[0]))
        print("Generation time average: %f" % average_accuracy_n_time[1])
        print('-' * 80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks, time_weight)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: (optimizer.fitness(x, time_weight)), reverse=True)

    # Print out the top 5 networks.
    print_top_networks(networks[:5])


def main():
    #Evolve a network
    generations = 3  # Number of times to evolve the population.
    population = 10  # Number of networks in each generation.
    # Allows you to select from 0 to 1 how much you want to weight the speed of the neural network training vs the accuracy
    time_weight = 0
    # Network parameters, network_params[0]: tuple of layer ranges, network_params[1]: tuple of nodes per layer
    network_params = [(0, 9), (1, 100)]
    X, y = load_breast_cancer(return_X_y=True)
    print("***Evolving %d generations with population %d***" %
          (generations, population))

    generate(generations, population, X, y, time_weight, network_params)

main()
