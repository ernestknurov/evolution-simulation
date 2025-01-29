import sys
import time
import copy
import pygame
import numpy as np
from pygame.locals import *
from collections import defaultdict

pygame.init()
pygame.font.init() 

HEIGHT, WIDTH = 740, 740
MARGIN = 50
CELL_SIZE = 10
GRID_SIZE = 64
POPULATION_SIZE = 500
STEPS_PER_GENERATION = 200
NUM_GENERATIONS = 100

screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(Color('white'))
pygame.display.set_caption("Game")

font = pygame.font.SysFont('notosansmono', 30)
clock = pygame.time.Clock()

# WARNING
# Location uses indices from numpy array, so the direction differ from that in pygame
direction_to_vector = {
    0: np.array([0, 1]), # right
    1: np.array([0, -1]), # left
    2: np.array([-1, 0]), # up
    3: np.array([1, 0]), # down
}

class Grid:
    def __init__(self):
        self.grid = self._create_grid()

    def _create_grid(self):
        grid = []
        for i in range(GRID_SIZE):
            grid.append([])
            for j in range(GRID_SIZE):
                rect = pygame.Rect((MARGIN + i * CELL_SIZE, MARGIN + j * CELL_SIZE), (CELL_SIZE, CELL_SIZE))
                grid[i].append(rect)
        return grid
    
    def render(self, surface):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pygame.draw.rect(surface, Color('gray'), self.grid[i][j], width=1)


class Creature:
    def __init__(self, location, genome):
        self.location = location
        self.genome = genome # 3 nums from 0 to 255
        self.center = (location[1] * CELL_SIZE + MARGIN + CELL_SIZE // 2, 
                       location[0] * CELL_SIZE + MARGIN + CELL_SIZE // 2)
        self.color = self._genome_to_color(genome)
        self.move_probs = self._genome_to_move_probs(genome)
        self.chance_to_move = self._genome_to_move_chance(genome)

    def _genome_to_color(self, genome):
        return genome
    
    def _genome_to_move_probs(self, genome):
        g1, g2, g3 = genome
        p1 = (g1 * 0.5 + g2 * 0.1 + g3 * 0.4) / 255.0
        p2 = (g1 * 0.2 + g2 * 0.2 + g3 * 0.6) / 255.0
        p3 = (g1 * 0.4 + g2 * 0.3 + g3 * 0.3) / 255.0
        p4 = (g1 * 0.3 + g2 * 0.5 + g3 * 0.2) / 255.0
        probs = [p1, p2, p3, p4]
        softmax_probs = np.exp(probs)/np.sum(np.exp(probs))
        return softmax_probs
    
    def _genome_to_move_chance(self, genome):
        g1, g2, g3 = genome 
        weighted_sum = g1 * 0.5 + g2 * 0.3 + g3 * 0.2  # Weights sum to 1 for stability
        probability = weighted_sum / 255.0  # Normalize to [0, 1]
        return probability
        
    def set_location(self, location):
        self.location = np.clip(location, 0, GRID_SIZE - 1)
        self.center = (self.location[1] * CELL_SIZE + MARGIN + CELL_SIZE // 2, 
                       self.location[0] * CELL_SIZE + MARGIN + CELL_SIZE // 2)
        
    def move(self, direction):
        self.set_location(self.location + direction)

    def render(self, surface):
        pygame.draw.circle(surface, center=self.center, radius = CELL_SIZE // 2, color=self.color)

    def update(self):
        if np.random.rand() < self.chance_to_move: 
            direction = np.random.choice(4, p=self.move_probs)
            self.move(direction_to_vector[direction])


class Population:
    def __init__(self, size=10):
        self.size = size
        self.collision_count = 0
        self.grid = - np.ones((GRID_SIZE, GRID_SIZE), dtype=np.int16) # -1 means the cell is free
        self.free_locations = np.argwhere(self.grid == -1)
        self._create_initial_population()
    
    def _create_initial_population(self):
        self.population = []
        for i in range(self.size):
            location = self._get_location_without_collisions()
            genome = np.random.choice(255, 3)
            creature = Creature(location, genome)
            self.population.append(creature)
            self.grid[*location] = i
    
    def _get_location_without_collisions(self):
        location_idx = np.random.choice(self.free_locations.shape[0])
        location = self.free_locations[location_idx]
        self.free_locations = self.free_locations[~np.all(self.free_locations == location, axis=1)]
        return location

    def select(self, select_criteria):
        before = len(self.population)
        self.population = list(filter(select_criteria, self.population))
        after = len(self.population)
        return np.round(after / before, 3) * 100

    def crossover(self):
        old_population = self.population
        self.population = []
        self.grid.fill(-1)
        self.free_locations = np.argwhere(self.grid == -1)
        parent_creature_ids = np.random.choice(len(old_population), self.size, replace=True)
        for i in range(self.size):
            location = self._get_location_without_collisions()
            parent_creature = old_population[parent_creature_ids[i]]
            genome = copy.deepcopy(parent_creature.genome)
            child_creature = Creature(location, genome)
            self.population.append(child_creature)
            self.grid[*location] = i


    def render(self, surface):
        for creature in self.population:
            creature.render(surface)
        
    def update(self):
        # RULES
        # 1. all creatures make a move simultaneously
        # 2. creature cannot move into occupied cells (cell was occupied and the moment of decision making)
        # 3. If 2+ creature wants to move to the same free cell, only 1 achieves it with uniform probability (this behavior can be adjusted later)

        old_locations = [creature.location for creature in self.population]

        # Rule 2
        for i, creature in enumerate(self.population):
            creature.update()
            if self.grid[*creature.location] != -1:
                creature.set_location(old_locations[i])

        # Rule 3
        location_dict = defaultdict(list)
        # create dict with location (key) and idx of creatures that have this location (value)
        for idx, creature in enumerate(self.population):
            location_dict[tuple(creature.location)].append(idx)
        
        for loc, indices in location_dict.items():
            if len(indices) > 1: # collision: one location with multiple creatures
                chosen = np.random.choice(indices)
                for idx in indices:
                    self.population[idx].set_location(old_locations[idx] if idx != chosen else loc)

        # Update the grid
        self.grid.fill(-1) # -1 means the cell is free
        for i, creature in enumerate(self.population):
            self.grid[*creature.location] = i

def select_criteria(creature):
    return creature.location[1] >= GRID_SIZE // 2


grid = Grid()
population = Population(POPULATION_SIZE)
text = font.render(f'Gen 0: survival rate - 100%', False, (0, 0, 0))
times = np.zeros(NUM_GENERATIONS * STEPS_PER_GENERATION)
step = 0
gen = 0
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            non_zero_times = times[times > 0]
            mean = non_zero_times.mean()
            print("Mean update time:", mean)
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            pass

    if gen == NUM_GENERATIONS:
        # make a pause
        while True: time.sleep(1.0)
    
    elif step == STEPS_PER_GENERATION:
        survival_rate = population.select(select_criteria)
        start_time = time.time()
        population.crossover()
        end_time = time.time()
        print("Crossover time:", end_time - start_time)
        gen += 1
        step = 0
        text = font.render(f'Gen {gen}: survival rate - {survival_rate}%', False, (0, 0, 0))

    else:
        start_time = time.time()
        population.update()
        end_time = time.time()
        times[gen * STEPS_PER_GENERATION + step] = end_time - start_time
        step += 1


    screen.fill(Color('white'))
    grid.render(screen)
    population.render(screen)
    screen.blit(text, (MARGIN, 0))
    pygame.display.update()
    clock.tick(1000)
