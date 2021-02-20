from datetime import datetime
from pathlib import Path
import math
import numpy as np
import os
import pickle
import random
import warnings

from helper import get_image, RGB2HEX, create_image_tiles, create_video, plot_population, is_in_rect
from image import get_dominant_color
from geo import get_haversine_distance, is_in_shape, get_cities, get_map_shape, create_grid_elements, get_optimized_tile_meta
from data import compressed_pickle, decompress_pickle


warnings.simplefilter(action='ignore', category=FutureWarning)
debug = False
initialize_data = False


def get_closest_city(point, cities):
    city_distances = []

    for city in cities:
        distance = get_haversine_distance(
            point[0], point[1], city['coords'][0], city['coords'][1])

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

    # Add closest city distance of chromosome for fitness evaluation
    current_population_closest_distances.append(closest_city['city_distance'])

    return closest_city['city_distance']


def evaluate_vegetation(point, tile):
    tile_meta = get_optimized_tile_meta(point, tile)

    # Assume that no vegetationindex or no data means there is no vegetation -> which is good
    if tile_meta.empty:
        return 1

    return 1 - (float(tile_meta['vegetationpercentage']) / 100)


def get_elevation_from_coords(point):
    # decide in which file to search
    tile_a1 = {'bbox': [-180, 0, -90, 90], 'tif': 'heightmap_A1.tif'}
    tile_a2 = {'bbox': [-180, -90, 90, 0], 'tif': 'heightmap_A2.tif'}
    tile_b1 = {'bbox': [90, 0, 0, 90], 'tif': 'heightmap_B1.tif'}
    tile_b2 = {'bbox': [-90, -90, 0, 0], 'tif': 'heightmap_B2.tif'}
    tile_c1 = {'bbox': [0, 0, 90, 90], 'tif': 'heightmap_C1.tif'}
    tile_c2 = {'bbox': [0, -90, 90, 0], 'tif': 'heightmap_C2.tif'}
    tile_d1 = {'bbox': [90, 0, 180, 90], 'tif': 'heightmap_D1.tif'}
    tile_d2 = {'bbox': [90, -90, 180, 0], 'tif': 'heightmap_D2.tif'}

    tiles = [tile_a1, tile_a2, tile_b1, tile_b2,
             tile_c1, tile_c2, tile_d1, tile_d2]

    src = ''
    result = 0

    for tile in tiles:
        if is_in_rect(point, tile['bbox']):
            tif = tile['tif']
            src = f'./image_data/{tif}'

    if src != '':
        result = os.popen(
            f'gdallocationinfo -valonly -b 1 -geoloc -wgs84 {src} {point[0]} {point[1]}').read()

    # Add elevation of chromosome for fitness evaluation
    current_population_highest_elevations.append(result)

    if debug:
        print(f'Elevation for {point[0]},{point[1]}: {result}')

    return float(result)


# -------------------------- GA Code
def get_constraint_data(point):
    grid_element_key = str(math.floor(
        point[0])) + "," + str(math.floor(point[1]))

    # evaluate by distance
    print(f'{str(datetime.now())} Calc constraint: city')
    distance = get_closest_city(point, cities)

    # 0 = point is the water (bad), 1 = not in water (good)
    print(f'{str(datetime.now())} Calc constraint: ocean')
    in_ocean = is_in_shape(point, ocean_grid_elements[grid_element_key])

    # If vegetationpercentage is high, give it a bad fitness
    print(f'{str(datetime.now())} Calc constraint: vegetation')
    vegetation = evaluate_vegetation(
        point, sentinel_row_tiles[grid_element_key])

    # Elevation from coord. ATM heigher = better?
    print(f'{str(datetime.now())} Calc constraint: city')
    elevation = get_elevation_from_coords(point)

    return {
        'constraint_ocean': in_ocean,
        'constraint_distance': distance,
        'constraint_vegetation': vegetation,
        'constraint_elevation': elevation
    }


def generate_initial_population(individuals, chromosome_length):
    pop_size = (individuals, chromosome_length)  # structure of array
    initial_population = np.random.uniform(low=-90, high=90, size=pop_size)
    initial_population[:, 0] *= 2

    if debug:
        print("initial population:")
        print(initial_population)

    return initial_population


def select_elite_chromosomes(population, scores, amount=50):
    highest_scoring_chromosomes = np.empty((amount, 2), dtype=np.float32)
    highest_score_indexes = np.argsort(-1*scores, axis=None)[:amount]

    for index in range(0, amount - 1):
        highest_scoring_chromosomes[index] = population[highest_score_indexes[index]]

    return highest_scoring_chromosomes


def select_individual_by_tournament(population, scores):
    # Get population size
    population_size = len(scores)

    # Pick individuals for tournament
    fighter_1 = random.randint(0, population_size - 1)
    fighter_2 = random.randint(0, population_size - 1)

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
    crossover_point = random.randint(1, chromosome_length - 1)

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


# -------------------------- GA
print(f'{str(datetime.now())} Start Atomic Tomb Finder')
# Info:
# GeoJson: [lng, lat]
# x y ~ lng lat

# -------------------------- GA parameter
# Note: population_size & max_elite_chromosomes have to be even amounts
population_size = 1000  # amount of coords
chromosome_length = 2  # coords (lon, lat)
max_elite_chromosomes = 50  # chromosomes to be taken into the next generation
max_generations = 50  # amount of iterations
mutation_rate = 0.002  # Rate of how the new population will be mutated

population = generate_initial_population(
    population_size, chromosome_length)
population_history = [population]
current_population_closest_distances = []
current_population_highest_elevations = []


# -------------------------- Load data
cities_file = 'geodata/cities_pop_5000000.geojson'
cities = get_cities(cities_file)

ocean_tile_file = 'geodata/processed_map_elements.pkl'
with open(ocean_tile_file, 'rb') as fp:
    ocean_grid_elements = pickle.load(fp)

height_map_file = './image_data/heightmap.png'
height_tiles = create_image_tiles(height_map_file)

sentinel_data_file = './geodata/sentinel_2_2020_07_01_to_2020_08_31_compressed'
sentinel_row_tiles = {}
for row_index in range(-90, 90):
    print(f'Loading row {row_index} into dict')
    file_path = f'{sentinel_data_file}/row_{row_index}.pbz2'
    data = decompress_pickle(file_path)

    sentinel_row_tiles.update(data)


# -------------------------- Execution
for generation in range(max_generations):
    print(f'{str(datetime.now())} Generation Nr.{str(generation)}')

    # get the fitnesses of all chromosomes in generation
    constraint_raw_data = []
    fitness = np.empty((population_size, 1))

    for chromosome_index in range(0, population_size - 1):
        constraint_raw_data.append(get_constraint_data(
            population[chromosome_index]))

    # Get distance delta form percentage transformation
    max_city_distance = np.max(np.asarray(
        current_population_closest_distances))
    min_city_distance = np.min(np.asarray(
        current_population_closest_distances))
    delta = max_city_distance - min_city_distance

    max_elevation = np.max(np.asarray(
        current_population_highest_elevations))

    # Calculate combined fitness
    # Simple: all fitnesses summed and divided by amount of constraints
    # TODO: weighted fitness function
    for chromosome_index in range(0, population_size - 1):
        el = constraint_raw_data[chromosome_index]
        constraintCount = len(el)

        # Transform min_city_distance and max_city_distance to 0 - 1 (percentage)
        fitness_distance = el['constraint_distance'] * \
            100 / max_city_distance / 100

        # Fitness for ocean check 1 or 0
        fitness_ocean = el['constraint_ocean']

        # Fitness for health/vegetation
        fitness_vegetation = el['constraint_vegetation']

        # Transform max_elevation to 0 - 1 (percentage)
        fitness_vegetation = el['constraint_vegetation'] * \
            100 / max_elevation / 100

        fitness[chromosome_index] = (
            fitness_distance + fitness_ocean + fitness_vegetation) / constraintCount

    max_fitness = np.max(fitness)
    max_fit_index = np.where(fitness == np.max(fitness))[0][0]

    if debug:
        print("fitness: ")
        print(fitness)
        print("max_fitness:")
        print(np.max(fitness))

    # Create an empty list for new population
    new_population = []
    current_population_closest_distances = []
    current_population_highest_elevations = []

    elite_chromosomes = select_elite_chromosomes(
        population, fitness, amount=max_elite_chromosomes)

    # Create new popualtion generating two children at a time
    # only create children of the amount of: population_size - max_elite_chromosomes children
    for i in range(int((population_size - max_elite_chromosomes) / 2)):
        parent_1 = select_individual_by_tournament(population, fitness)
        parent_2 = select_individual_by_tournament(population, fitness)

        child_1, child_2 = breed_by_crossover(parent_1, parent_2)

        new_population.append(child_1)
        new_population.append(child_2)

    # Replace the old population with the new one
    population = np.concatenate(
        (np.array(new_population), elite_chromosomes))

    # Apply mutation
    population = randomly_mutate_population(population, mutation_rate)
    population_history.append(population)

    if debug:
        print("population:")
        print(population)


print("==================================================")
print("Result")
print(f"Best chromosome: {str(population[max_fit_index])}")
print(f"Fitness: {str(max_fitness)}")
print("https://www.google.com/maps/search/?api=1&query=" +
      str(population[max_fit_index][1]) + "," + str(population[max_fit_index][0]))
print("==================================================")
print("Data")
print(f"Cities: {cities_file}")
print(f"Ocean Data: {ocean_tile_file}")
print(f"Sentinel Data: {sentinel_data_file}")
print("...")

# [-160.42666633  -47.61328754] - new
# [-160.58738847  -47.35714332] - maurice
# Best chromosome: [150.38105895 -79.62562572]

print("==================================================")
print("Create Video")
top_points_indexes = np.argsort(-1*fitness, axis=None)[:10]
top_points_fitness = fitness[top_points_indexes].flatten()
top_combined = np.vstack((top_points_indexes, top_points_fitness))

create_video(population_history, top_combined)

print("==================================================")
print("Show last population")
top_points_indexes = np.argsort(-1*fitness, axis=None)[:len(fitness) - 1]
top_points_fitness = fitness[top_points_indexes].flatten()
top_combined = np.vstack((top_points_indexes, top_points_fitness))

plot_population(population, top_combined)
