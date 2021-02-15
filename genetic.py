from datetime import datetime
from pathlib import Path
import math
import numpy as np
import os
import pickle
import random

from helper import get_image, RGB2HEX, create_image_tiles, create_video, plot_population
from image import get_dominant_color
from geo import get_haversine_distance, is_in_shape, get_elevation_tile_from_web, get_cities, get_map_shape, create_grid_elements


# Info:
# GeoJson: [lng, lat]
# x y ~ lng lat

debug = False


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


# -------------------------- GA Code
def get_constraint_data(point):
    if point[0] > 180 or point[0] < -180:
        print(point)
    grid_element_key = str(math.floor(
        point[0])) + "," + str(math.floor(point[1]))

    # evaluate by distance
    distance = get_closest_city(point, cities)

    # 0 = point is the water (bad), 1 = not in water (good)
    in_ocean = is_in_shape(point, grid_elements[grid_element_key])

    return {
        'constraint_ocean': in_ocean,
        'constraint_distance': distance,
        'constraint_height': 0,
        'constraint_health': 0,
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


# -------------------------- Load data
cities_file = 'geodata/cities_pop_5000000.geojson'
cities = get_cities(cities_file)
bedrock = 'ufff'


# -------------------------- Prepare data
pkl_file = Path("geodata/processed_map_elements.pkl")

with open('geodata/processed_map_elements.pkl', 'rb') as fp:
    grid_elements = pickle.load(fp)


# -------------------------- GA parameter
# Note: population_size & max_elite_chromosomes have to be even amounts
population_size = 1000  # amount of coords
chromosome_length = 2  # coords (lon, lat)
max_elite_chromosomes = 50  # chromosomes to be taken into the next generation
max_generations = 50  # amount of iterations
mutation_rate = 0.002  # Rate of how the new population will be mutated


population = generate_initial_population(population_size, chromosome_length)
population_history = [population]
current_population_closest_distances = []

# get_image_data()
# tiles = create_image_tiles('./image_data/heightmap.png')

# test_img = tiles['-122,41']
# cv2.imshow('image', test_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # img = tiles['13,0']


# elevation_file = get_elevation_tile_from_web(
#     (-122, 41, -121, 42), 'image_data/-122_41_DEM.tif')
# gtif = gdal.Open(elevation_file)
# srcband = gtif.GetRasterBand(1)

# # Get raster statistics
# stats = srcband.GetStatistics(True, True)

# # Print the min, max, mean, stdev based on stats index
# print("[ STATS ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % (
#     stats[0], stats[1], stats[2], stats[3]))
# test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)


# modified_image = test_img.reshape(
#     test_img.shape[0]*test_img.shape[1], 1)

# clf = KMeans(n_clusters=4)
# labels = clf.fit_predict(modified_image)

# counts = Counter(labels)

# center_colors = clf.cluster_centers_
# # We get ordered colors by iterating through the keys
# ordered_colors = [center_colors[i] for i in counts.keys()]
# hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
# rgb_colors = [ordered_colors[i] for i in counts.keys()]

# plt.figure(figsize=(8, 6))
# plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)

# plt.show()

# exit()

# -------------------------- Execution
for generation in range(max_generations):
    print(str(datetime.now()) + " Generation #" + str(generation))

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

    # Calculate combined fitness
    # Simple: all fitnesses summed and divided by amount of constraints
    # TODO: weighted fitness function
    for chromosome_index in range(0, population_size - 1):
        el = constraint_raw_data[chromosome_index]
        # Transform min_city_distance and max_city_distance to 0 - 1
        fitness_distance = el['constraint_distance'] * \
            100 / max_city_distance / 100

        # Fitness for ocean check 1 or 0
        fitness_ocean = el['constraint_ocean']

        fitness[chromosome_index] = (
            fitness_distance + fitness_ocean) / 2

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
    population = np.concatenate((np.array(new_population), elite_chromosomes))

    # Apply mutation
    population = randomly_mutate_population(population, mutation_rate)
    population_history.append(population)

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

print("==================================================")
print("Create Video")
top_points_indexes = np.argsort(-1*fitness, axis=None)[:10]
top_points_fitness = fitness[top_points_indexes].flatten()
top_combined = np.vstack((top_points_indexes, top_points_fitness))

# create_video(population_history, top_combined)

print("==================================================")
print("Show last population")
top_points_indexes = np.argsort(-1*fitness, axis=None)[:len(fitness) - 1]
top_points_fitness = fitness[top_points_indexes].flatten()
top_combined = np.vstack((top_points_indexes, top_points_fitness))

plot_population(population, top_combined)
