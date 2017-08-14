from random import randint
from sklearn.neural_network import MLPClassifier


from functools import reduce
from operator import add
import random


class Optimizer():
    def __init__(self, retain=0.4, random_select=0.1, mutate_chance=0.2):
        # Create an optimizer
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain

    def randomize_hidden_layers(self, network_params):
        # create a random sized number of hidden layers
        hidden_lst = []
        for i in range(0, randint(network_params[0][0], network_params[0][1])):
            hidden_lst.append(randint(network_params[1][0], network_params[1][1]))
        return tuple(hidden_lst)

    def create_population(self, count, network_params):
        # Create a population of random networks
        pop = []
        for _ in range(0, count):
            # Create a random network.
            network = MLPClassifier(hidden_layer_sizes=self.randomize_hidden_layers(network_params), solver='sgd',
                                    learning_rate='adaptive', max_iter=1000)

            # Add the network to our population.
            pop.append([network, 0])

        return pop

    def fitness(self, network, time_weight):
        return ((1 - network[0].__getattribute__('loss_')) * (1 - time_weight)) + (1 - network[1] * time_weight)

    def grade(self, pop, time_weight):
        # Find average fitness for a population
        summed = reduce(add, (self.fitness(network, time_weight) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        # make two children as parts of their parents
        children = []
        for _ in range(2):
            hidden_layers = []
            # Loop through the parameters and pick params for the kid.
            # randomly pick the number of hidden layers the kid will have
            num_of_layers = random.choice([len(mother[0].__getattribute__('coefs_')), len(father[0].__getattribute__('coefs_'))])
            for i in range(0, num_of_layers - 1):

                # randomly pick number of nodes from one of the mother's columns (also chosen at random)
                mother_num_nodes = len(mother[0].__getattribute__('coefs_')[random.choice([0, len(mother[0].__getattribute__('coefs_')) - 1])])

                # randomly pick number of nodes from one of the fathers's columns (also chosen at random)
                father_num_nodes = len(father[0].__getattribute__('coefs_')[random.choice([0, len(father[0].__getattribute__('coefs_')) - 1])])

                hidden_layers.append(random.choice([mother_num_nodes, father_num_nodes]))

            # Now create a network object.
            network = MLPClassifier(hidden_layer_sizes=tuple(hidden_layers), solver='sgd', learning_rate='adaptive')

            children.append([network, 0])
        return children

    def mutate(self, network):
        # Randomly mutate one part of the network
        # Choose a random key.
        mutation = random.choice(list(range(0, len(network[0].__getattribute__('coefs_')))))
        # Mutate one of the params.
        network[0].__getattribute__('coefs_')[mutation][random.choice(list(range(0, len(network[0].__getattribute__('coefs_')[mutation]))))] = random.randrange(-1, 1)

        return network

    def evolve(self, pop, time_weight):
        # Evolve a population of networks
        # Get scores for each network.
        graded = [(self.fitness(network, time_weight), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        # Check that there will be at least 2 to mate
        if (len(graded) * self.retain) >= 2:
            retain_length = int(len(graded) * self.retain)

        else:
            print("Population number is too small for evolution. Please change to a larger number and try again.")
            exit(0)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Randomly mutate some of the networks we're keeping.
        for individual in parents:
            if self.mutate_chance > random.random():
                individual = self.mutate(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)
        parents.extend(children)
        return parents
