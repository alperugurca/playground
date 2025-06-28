# Ant Colony Optimization
# Source: https://towardsdatascience.com/ant-colony-optimization-intuition-code-visualization-9412c369be81

import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

city_coordinates = [
    (5, 15),
    (17, 38),
    (28, 52),
    (32, 43),
    (23, 43),
    (2, 7),
    (69, 70),
    (24, 52),
    (21, 55),
    (87, 99)
]

class SimpleACO:
    def __init__(self, salesman, n_ants, rho):
        self.salesman = salesman
        self.n_ants = n_ants
        self.rho = rho
        self.pheromone = 0.01*np.ones((self.salesman.num_cities, self.salesman.num_cities))

    def update_map(self, new_salesman):
        num_old_cities = len(self.pheromone)
        retain = deepcopy(self.pheromone)
        self.salesman = new_salesman
        self.pheromone = 0.01*np.ones((self.salesman.num_cities, self.salesman.num_cities))
        self.pheromone[:num_old_cities,:num_old_cities] = retain

    def run(self, n_iterations):
        best_distance = float('inf')
        best_solution = None

        for _ in tqdm(range(n_iterations)):
            solutions = [self.generate_path() for _ in range(self.n_ants)]
            self.update_pheromone(solutions)

            for solution in solutions:
                distance = -self.salesman.fitness(solution)
                if distance < best_distance:
                    best_distance = distance
                    best_solution = solution

        return best_solution

    def generate_path(self):
        path = [random.randint(0, self.salesman.num_cities - 1)]
        while len(path) < self.salesman.num_cities:
            next_city = self.choose_next_city(path[-1], path)
            path.append(next_city)
        path.append(path[0])
        return path

    def update_pheromone(self, solutions):
        self.pheromone *= (1 - self.rho)
        for solution in solutions:
            for i in range(len(solution) - 1):
                self.pheromone[solution[i]][solution[i+1]] += 1 / -self.salesman.fitness(solution)

    def choose_next_city(self, current_city, path):
        probabilities = []
        for city in range(self.salesman.num_cities):
            if city not in path:
                probabilities.append(self.pheromone[current_city][city])
            else:
                probabilities.append(0)

        probabilities = np.array(probabilities)/np.sum(probabilities)
        next_city = np.random.choice(range(self.salesman.num_cities), p=probabilities)
        return next_city
class Salesman:
    def __init__(self, num_cities, x_lim, y_lim, read_from_txt=None):
        if read_from_txt:
            self.city_locations = []
            f = open(read_from_txt)
            for i, line in enumerate(f.readlines()):
                if i==num_cities:
                    break
                node_val = line.split()
                self.city_locations.append(
                    (float(node_val[-2]), float(node_val[-1]))
                )
            self.num_cities = len(self.city_locations)
            self.x_lim = np.max(np.array(self.city_locations)[:,0])
            self.y_lim = np.max(np.array(self.city_locations)[:,1])

        else:   # generate randomly
            self.num_cities = num_cities
            self.x_lim = x_lim
            self.y_lim = y_lim
            x_loc = np.random.uniform(0, x_lim, size=num_cities)
            y_loc = np.random.uniform(0, y_lim, size=num_cities)
            self.city_locations = [
                (x,y) for x,y in zip(x_loc,y_loc)
            ]
        self.distances = self.calculate_distances()

    def calculate_distances(self):
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                dist = np.sqrt((self.city_locations[i][0] - self.city_locations[j][0]) ** 2 + (self.city_locations[i][1] - self.city_locations[j][1]) ** 2)
                distances[i][j] = distances[j][i] = dist
        return distances

    def fitness(self, solution):
        total_distance = 0
        for i in range(self.num_cities - 1):
            total_distance += self.distances[solution[i]][solution[i+1]]
        total_distance += self.distances[solution[-1]][solution[0]]   ###
        fitness = -total_distance
        return fitness

    def visualize(self, solution, save_id=None):
        n = len(solution)
        assert n == len(self.city_locations), 'The solution must correspond to all cities'
        for i, (x,y) in enumerate(self.city_locations):
            plt.plot(x, y, "ro")
            plt.annotate(i, (x, y))

        ordered_cities = [self.city_locations[idx] for idx in solution]
        x_coord = [x for (x,y) in  ordered_cities] + [ordered_cities[0][0]]  ###
        y_coord = [y for (x,y) in  ordered_cities] + [ordered_cities[0][1]]  ###
        distance = -self.fitness(solution)

        plt.plot(x_coord, y_coord, "gray")
        plt.title("Connected cities (%.1f) according to solution" % distance)
        if save_id is not None:
            filename = "results/plot_%03d.png" % save_id
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @classmethod
    def from_coordinates(cls, coordinates):
        instance = cls(num_cities=len(coordinates), x_lim=100, y_lim=100)
        instance.city_locations = coordinates
        instance.num_cities = len(coordinates)
        instance.x_lim = np.max(np.array(coordinates)[:,0])
        instance.y_lim = np.max(np.array(coordinates)[:,1])
        instance.distances = instance.calculate_distances()
        return instance


salesman = Salesman.from_coordinates(city_coordinates)
aco = SimpleACO(salesman, n_ants=20, rho=0.1)
best_solution = aco.run(n_iterations=2000)
salesman.visualize(best_solution[:-1])

# İkinci çalıştırma için
new_salesman = Salesman.from_coordinates(city_coordinates)
aco.update_map(new_salesman)
new_best_solution = aco.run(n_iterations=500)
new_salesman.visualize(new_best_solution[:-1])