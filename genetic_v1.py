import matplotlib.pyplot as plt
import json
import numpy as np
import math
import fiona
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.geometry.polygon import LinearRing
from shapely.ops import split
import random
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import os
import sys
from datetime import datetime

# Info:
# GeoJson: [lng, lat]
# x y ~ lng lat

debug = False


def get_cities(filename='geodata/cities_pop_10000000.geojson'):
    with open(filename) as f:
        gj = json.load(f)
        cities = [{'name': feat['properties']['name'], 'coords': [float(feat['geometry']['coordinates'][0]), float(
            feat['geometry']['coordinates'][1])]} for feat in gj['features']]

    print("Loaded " + str(len(cities)) + " city coordinates")

    return cities


def get_map_shape(filename='geodata/ne_10m_ocean/ne_10m_ocean.shp', simplify=False, tolerance=0.5):
    multipolygon = MultiPolygon([shape(pol['geometry'])
                                 for pol in fiona.open(filename)])

    print("Loaded " + str(filename) + " multipolygon")

    if not multipolygon.is_valid:
        print("Shape not valid. Applying fix")
        multipolygon = multipolygon.buffer(0)

    if simplify:
        simplified = multipolygon.simplify(tolerance)
        simplified = simplified.buffer(0)

        return simplified

    return multipolygon


def create_grid_elements(shape):
    longitude_size = 0.99999999
    latitude_size = 0.99999999
    grid_dict = {}

    print("Start grid creation (this may take a while)")

    for latitude in range(-90, 90):
        for longitude in range(-180, 180):
            grid_element = Polygon([
                (longitude, latitude),
                (longitude + longitude_size, latitude),
                (longitude + longitude_size, latitude + latitude_size),
                (longitude, latitude + latitude_size)
            ])

            splitted = grid_element.intersection(shape)

            grid_dict[str(longitude) + ',' + str(latitude)] = splitted

        print(str(datetime.now()) +
              " Calculated intersections for lat row " + str(latitude) + "/90")

    return grid_dict


def get_haversine_distance(lon1, lat1, lon2, lat2):
    R = 6378.137
    R = 6371
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * \
        math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c

    # Kilometers
    return d


def get_closest_city(point, cities):
    city_distances = []

    for city in cities:
        distance = get_haversine_distance(
            point[0], point[1], city['coords'][0], city['coords'][1])
        # distance = get_distance(point, city['coords'])

        city_distances.append({
            'city_coords': city['coords'],
            'city_distance': distance,
            'city_name': city['name']
        })

        if (debug):
            coords = city['coords']
            name = city['name']
            print(
                f'Distance from {name} (lon/lat {point[0]:.7f}/{point[1]:.7f}) to lon/lat {coords[0]:.7f}/{coords[1]:.7f}: {distance}')

    sorted_city_distances = sorted(
        city_distances, key=lambda x: x['city_distance'], reverse=False)

    closest_city = sorted_city_distances[0]

    return closest_city['city_distance']


def is_in_shape(point, shape):
    p1 = Point(point)

    return p1.within(shape)

# -------------------------- GA Code


def get_fitness(point):
    grid_element_key = str(math.floor(
        point[0])) + "," + str(math.floor(point[1]))
    # evaluate by distance
    fitness_distance = get_closest_city(point, cities)
    fitness_ocean = is_in_shape(point, grid_elements[grid_element_key])

    if fitness_ocean:
        # when point is in ocean return really bad fitness score
        return -999999

    return fitness_distance


def generate_initial_population(individuals, chromosome_length):
    pop_size = (individuals, chromosome_length)  # structure of array
    initial_population = np.random.uniform(low=-90, high=90, size=pop_size)
    initial_population[:, 0] *= 2

    print("initial population:")
    print(initial_population)

    return initial_population


def select_elite_chromosome(population, scores):
    # TODO: get n top performers of the current generation
    return [0, 0]


def select_individual_by_tournament(population, scores):
    # Get population size
    population_size = len(scores)

    # Pick individuals for tournament
    fighter_1 = random.randint(0, population_size-1)
    fighter_2 = random.randint(0, population_size-1)

    # Get fitness score for each
    fighter_1_fitness = scores[fighter_1]
    fighter_2_fitness = scores[fighter_2]

    # Identify undividual with highest fitness
    # Fighter 1 will win if score are equal
    if fighter_1_fitness >= fighter_2_fitness:
        winner = fighter_1
    else:
        winner = fighter_2

    # Return the chromsome of the winner
    return population[winner, :]


def breed_by_crossover(parent_1, parent_2):
    # Get length of chromosome
    chromosome_length = len(parent_1)

    # Pick crossover point, avoding ends of chromsome
    crossover_point = random.randint(1, chromosome_length-1)

    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point],
                         parent_2[crossover_point:]))

    child_2 = np.hstack((parent_2[0:crossover_point],
                         parent_1[crossover_point:]))

    # Return children
    return child_1, child_2


def randomly_mutate_population(population, mutation_probability):
    # Apply random mutation
    random_mutation_array = np.random.random_sample(
        size=(population.shape))

    random_mutation_boolean = \
        random_mutation_array <= mutation_probability

    population[random_mutation_boolean] = \
        np.logical_not(population[random_mutation_boolean])

    # Return mutation population
    return population


# -------------------------- Load data
cities_file = 'geodata/cities_pop_5000000.geojson'
cities = get_cities(cities_file)
bedrock = 'ufff'


# -------------------------- Prepare data
pkl_file = Path("geodata/processed_map_elements.pkl")

if not Path('geodata/').is_dir():
    os.mkdir('geodata/')

if not pkl_file.is_file():
    # land = get_map_shape('geodata/ne_10m_land/ne_10m_land.shp',
    #                      simplify=False, tolerance=5)
    oceans = get_map_shape(
        'geodata/ne_10m_ocean_scale_rank/ne_10m_ocean_scale_rank.shp', simplify=False, tolerance=0.5)
    print("Did not find processed map")
    grid_elements = create_grid_elements(oceans)

    with open('geodata/processed_map_elements.pkl', 'wb') as fp:
        pickle.dump(grid_elements, fp, pickle.HIGHEST_PROTOCOL)

with open('geodata/processed_map_elements.pkl', 'rb') as fp:
    grid_elements = pickle.load(fp)


# # POC for interectinal function
# splitted = grid_elements['8,4'].intersection(oceans)
# print(splitted)
#
# fig, axs = plt.subplots()
# axs.set_aspect('equal', 'datalim')
#
# for key in grid_elements:
#     el = grid_elements[key]
#
#     if not el.is_empty:
#         if el.geom_type == 'MultiPolygon':
#             for geom in el.geoms:
#                 xs, ys = geom.exterior.xy
#                 axs.fill(xs, ys, alpha=0.5, c=np.random.rand(3,))
#
#         if el.geom_type == 'Polygon':
#             xs, ys = el.exterior.xy
#             axs.fill(xs, ys, alpha=0.5, c=np.random.rand(3,))
#
# fig, axs = plt.subplots()
# axs.set_aspect('equal', 'datalim')
#
# for geom in land.geoms:
#     xs, ys = geom.exterior.xy
#     axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
#
# plt.show()


# -------------------------- GA parameter
population_size = 1000  # amount of coords
chromosome_length = 2  # coords (lon, lat)
max_generations = 50
mutation_rate = 0.002

population = generate_initial_population(population_size, chromosome_length)

# -------------------------- Execution
for generation in range(max_generations):
    print(str(datetime.now()) + " Generation #" + str(generation))

    # get the fitnesses of all chromosomes in generation
    fitness = np.empty((population_size, 1))
    for chromosome_index in range(0, population_size - 1):
        fitness[chromosome_index] = get_fitness(population[chromosome_index])

    max_fitness = np.max(fitness)
    max_fit_index = np.where(fitness == np.max(fitness))[0][0]

    if debug:
        print("fitness: ")
        print(fitness)
        print("max_fitness:")
        print(np.max(fitness))

    # Create an empty list for new population
    new_population = []

    # Create new popualtion generating two children at a time
    for i in range(int(population_size / 2)):
        parent_1 = select_individual_by_tournament(population, fitness)
        parent_2 = select_individual_by_tournament(population, fitness)

        child_1, child_2 = breed_by_crossover(parent_1, parent_2)

        new_population.append(child_1)
        new_population.append(child_2)

    # Replace the old population with the new one
    population = np.array(new_population)

    # Apply mutation
    population = randomly_mutate_population(population, mutation_rate)

    if debug:
        print("population:")
        print(population)

best_result = population[max_fit_index]

print("==================================================")
print("Result")
print("Best chromosome: " + str(best_result))
print("Fitness: " + str(max_fitness))
print("https://www.google.com/maps/search/?api=1&query=" +
      str(best_result[1]) + "," + str(best_result[0]))
print("==================================================")
print("Data")
print("Cities: " + cities_file)
print("...")


# [-160.42666633  -47.61328754] - new
# [-160.58738847  -47.35714332] - maurice
# Best chromosome: [150.38105895 -79.62562572]
