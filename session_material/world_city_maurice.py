import math
import numpy
import json

import random


def get_distance(coord1, coord2):  # coord = (lon, lat) , london = (-0.118092, 51.509865)
    dlon = math.radians(coord2[0]) - math.radians(coord1[0])
    dlat = math.radians(coord2[1]) - math.radians(coord1[1])
    a = (math.sin(dlat/2))**2 + math.cos(math.radians(coord1[1]))*math.cos(
        math.radians(coord2[1]))*(math.sin(dlon/2))**2
    distance = 6371*2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return distance


def get_fitness(coord):
    min_dist = 999999999999999999999
    i = 0
    for cities in data['features']:
        dist = get_distance(
            coord, data['features'][i]['geometry']['coordinates'])
        if dist < min_dist:
            min_dist = dist
        i = i + 1
    return min_dist


# load json
with open('geodata/cities_pop_10000000.geojson', encoding="utf8") as f:
    data = json.load(f)


test = (-0.118, 51.510)
mindis = get_fitness(test)
print(mindis)


# define meta par
chr_per_pop = 1000
gen_per_chr = 2

num_generations = 50
num_parents = 500
num_offspring = 500
########################################################
mutations = 100

pop_size = (chr_per_pop, gen_per_chr)
offspring_size = (num_offspring, gen_per_chr)

# create initial population
new_population = numpy.random.uniform(low=-90, high=90, size=pop_size)
new_population[:, 0] *= 2
print("initial population:")
print(new_population)

# generations
for generation in range(num_generations):

    # get the fitnesses of all chromosomes in generation
    fitness = numpy.empty((pop_size[0], 1))
    for chromosome in range(0, pop_size[0]):
        fitness[chromosome] = get_fitness(new_population[chromosome])
    ##gen_fit = numpy.append(new_population, fitness, axis=1)
    print("fitness:")
    print(fitness)
    max_fitness = numpy.max(fitness)
    print("max_fitness:")
    print(numpy.max(fitness))

    # choose parents
    parents = numpy.empty((num_parents, pop_size[1]))
    for parent in range(num_parents):
        max_fit_index = numpy.where(fitness == numpy.max(fitness))
        max_fit_index = max_fit_index[0][0]
        parents[parent][0] = new_population[max_fit_index][0]
        parents[parent][1] = new_population[max_fit_index][1]
        fitness[max_fit_index] = -99999999
    print("parents:")
    print(parents)

    # create offspring (crossover + elite)
    offspring = numpy.empty(offspring_size)
    crossover_point = 1

    for child in range(offspring_size[0]):
        parent1_index = random.randrange(0, parents.shape[0])
        parent2_index = random.randrange(0, parents.shape[0])
        offspring[child, 0] = parents[parent1_index, 0]
        offspring[child, 1] = parents[parent2_index, 1]
    print("offspring:")
    print(offspring)

    # mutation
    for mut in range(mutations):
        mut_index = (random.randrange(
            offspring.shape[0]), random.randrange(offspring.shape[1]))
        if mut_index[1] == 0:  # longitude
            offspring[mut_index] = numpy.random.uniform(
                low=-180, high=180, size=1)
        if mut_index[1] == 1:  # latitude
            offspring[mut_index] = numpy.random.uniform(
                low=-90, high=90, size=1)
    print("mutated offspring:")
    print(offspring)

    # mutated offspring as new generation
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring
    print("new_population:")
    print(new_population)

    print("NEXT GENERATION")

# best
print("BEST:")
print(parents[0, :])
print(max_fitness)


"""
punkt = (-179, -39)
sydney = (151, -33)
dist = get_distance(punkt, sydney)
print(dist)
"""
