import random
from typing import List, Tuple, Optional
from pyamaze import maze, agent, COLOR
from math import sqrt
import time


class MazeGenerator:
    def __init__(self, rows: int, cols: int, loop_percent: int = 100):
        self.rows = rows
        self.cols = cols
        self.loop_percent = loop_percent
        self.maze = maze(rows, cols)

    def create_maze(self) -> maze:
        self.maze.CreateMaze(loopPercent=self.loop_percent)
        return self.maze


class Individual:
    def __init__(self, path: Optional[List[Tuple[int, int]]] = None):
        self.path = path if path is not None else []
        self.fitness: float = float('inf')

    def __len__(self):
        return len(self.path)

    def __repr__(self):
        return f"Individual(path_length={len(self.path)}, fitness={self.fitness})"


class GeneticAlgorithm:
    def __init__(
        self,
        rows: int,
        cols: int,
        population_size: int,
        generations: int,
        loop_percent: int = 100,
        elite_size: int = 2,
        mutation_rate: float = 0.2,
        penalty_factor: int = 20,
    ):
        self.rows = rows
        self.cols = cols
        self.population_size = population_size
        self.generations = generations
        self.loop_percent = loop_percent
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.penalty_factor = penalty_factor

        self.maze_generator = MazeGenerator(rows, cols, loop_percent)
        self.maze = self.maze_generator.create_maze()
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None

    def initialize_population(self):
        self.population = [
            self.generate_individual() for _ in range(self.population_size)
        ]

    def generate_individual(self) -> Individual:
        max_length = self.rows * self.cols * 2
        path = [(self.rows, self.cols)]
        visited = set(path)

        while path[-1] != (1, 1) and len(path) < max_length:
            current = path[-1]
            moves = self.get_possible_moves(current, visited)
            if moves:
                next_move = random.choice(moves)
                path.append(next_move)
                visited.add(next_move)
            else:
                break

        return Individual(path)

    def get_possible_moves(
        self, position: Tuple[int, int], visited: set
    ) -> List[Tuple[int, int]]:
        directions = ['N', 'S', 'E', 'W']
        deltas = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
        moves = []

        for direction in directions:
            if self.maze.maze_map[position][direction] == 1:
                delta = deltas[direction]
                next_pos = (position[0] + delta[0], position[1] + delta[1])
                if next_pos not in visited:
                    moves.append(next_pos)
        return moves

    def evaluate_fitness(self, individual: Individual, objective: Tuple[int, int] = (1, 1)):
        if individual.path[-1] == objective:
            individual.fitness = len(individual.path)
        else:
            distance = abs(individual.path[-1][0] - objective[0]) + abs(individual.path[-1][1] - objective[1])
            individual.fitness = len(individual.path) + (distance * self.penalty_factor)

    def evaluate_population(self):
        for individual in self.population:
            self.evaluate_fitness(individual, (1, 1))

    def select_tournament(self, tournament_size: int = 3) -> List[Individual]:
        selected = []
        for _ in range(self.population_size - self.elite_size):
            tournament = random.sample(self.population, tournament_size)
            winner = min(tournament, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        common_points = list(set(parent1.path) & set(parent2.path))
        if not common_points:
            return Individual(parent1.path.copy()), Individual(parent2.path.copy())

        crossover_point = random.choice(common_points)
        index_p1 = parent1.path.index(crossover_point)
        index_p2 = parent2.path.index(crossover_point)

        child1_path = parent1.path[:index_p1] + parent2.path[index_p2:]
        child2_path = parent2.path[:index_p2] + parent1.path[index_p1:]

        child1_path = self.eliminate_loops(child1_path)
        child2_path = self.eliminate_loops(child2_path)

        return Individual(child1_path), Individual(child2_path)

    def eliminate_loops(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        visited = set()
        new_path = []
        for cell in path:
            if cell in visited:
                break
            new_path.append(cell)
            visited.add(cell)

        while new_path[-1] != (1, 1):
            current = new_path[-1]
            moves = self.get_possible_moves(current, visited)
            if moves:
                next_move = random.choice(moves)
                new_path.append(next_move)
                visited.add(next_move)
            else:
                break
        return new_path

    def mutate(self, individual: Individual, num_mutations: int = 2):
        for _ in range(num_mutations):
            if len(individual.path) < 3:
                continue
            mutation_point = random.randint(1, len(individual.path) - 2)
            current = individual.path[mutation_point]
            visited = set(individual.path[:mutation_point])

            moves = self.get_possible_moves(current, visited)
            if moves:
                new_move = random.choice(moves)
                individual.path[mutation_point] = new_move
                individual.path = individual.path[:mutation_point + 1]
                while individual.path[-1] != (1, 1):
                    current = individual.path[-1]
                    possible_moves = self.get_possible_moves(current, set(individual.path))
                    if possible_moves:
                        next_move = random.choice(possible_moves)
                        individual.path.append(next_move)
                    else:
                        break

    def run(self) -> Optional[List[Tuple[int, int]]]:
        self.initialize_population()
        self.evaluate_population()
        self.best_individual = min(self.population, key=lambda ind: ind.fitness)

        for generation in range(1, self.generations + 1):
            self.population.sort(key=lambda ind: ind.fitness)
            current_best = self.population[0]

            if current_best.fitness < self.best_individual.fitness:
                self.best_individual = current_best

            if current_best.path[-1] == (1, 1) and current_best.fitness == len(current_best.path):
                print(f"Optimal solution found at generation {generation}")
                break

            elites = self.population[:self.elite_size]

            selected = self.select_tournament()

            next_generation = elites.copy()
            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)

                if random.random() < self.mutation_rate:
                    self.mutate(child1)
                if random.random() < self.mutation_rate:
                    self.mutate(child2)

                next_generation.extend([child1, child2])

            self.population = next_generation[:self.population_size]
            self.evaluate_population()

            if generation % 100 == 0 or generation == 1:
                print(f"Generation {generation}: Best fitness = {self.best_individual.fitness}")

        valid_solutions = [ind for ind in self.population if ind.path[-1] == (1, 1)]
        if valid_solutions:
            best_solution = min(valid_solutions, key=lambda ind: len(ind.path))
            self.best_individual = best_solution
            return best_solution.path
        else:
            print("No solution reaching the objective was found.")
            return None


class MazeSolver:
    def __init__(
        self,
        rows: int = 20,
        cols: int = 20,
        population_size: int = 100,
        generations: int = 1000,
        loop_percent: int = 100,
        elite_size: int = 2,
    ):
        self.rows = rows
        self.cols = cols
        self.population_size = population_size
        self.generations = generations
        self.loop_percent = loop_percent
        self.elite_size = elite_size

    def solve(self):
        ga = GeneticAlgorithm(
            rows=self.rows,
            cols=self.cols,
            population_size=self.population_size,
            generations=self.generations,
            loop_percent=self.loop_percent,
            elite_size=self.elite_size,
        )
        solution = ga.run()
        return ga.maze, solution


class MazeVisualizer:
    @staticmethod
    def display_optimal_path(maze_obj: maze, solution: Optional[List[Tuple[int, int]]]):
        if not solution:
            print("No solution to display.")
            return

        a_optimal = agent(
            maze_obj,
            x=maze_obj.rows,
            y=maze_obj.cols,
            footprints=True,
            color=COLOR.red,
            filled=True,
            shape='arrow'
        )

        maze_obj.tracePath({a_optimal: solution}, delay=0)

        maze_obj.run()


def main():
    filas = 20
    columnas = 20
    tamano_poblacion = 100
    generaciones = 1000
    num_elite = 2

    solver = MazeSolver(
        rows=filas,
        cols=columnas,
        population_size=tamano_poblacion,
        generations=generaciones,
        loop_percent=100,
        elite_size=num_elite,
    )
    laberinto, solucion = solver.solve()

    MazeVisualizer.display_optimal_path(laberinto, solucion)

    if solucion:
        print("Camino encontrado:", solucion)


if __name__ == "__main__":
    main()
