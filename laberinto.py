# Integrantes:
#  - Cristhian
#  - Chesney
#  - Marco
#  - Eros

from pyamaze import maze,agent,COLOR
import random

def generar_poblacion(filas, columnas, tamano_poblacion, inicio, meta):
    poblacion = []
    
    for _ in range(tamano_poblacion):
        camino = [inicio]
        posicion_actual = inicio
        while posicion_actual != meta:
            x, y = posicion_actual
            movimientos_posibles = []
            
            if x > 1:  # Arriba
                movimientos_posibles.append((x - 1, y))
            if x < filas:  # Abajo
                movimientos_posibles.append((x + 1, y))
            if y > 1:  # Izquierda
                movimientos_posibles.append((x, y - 1))
            if y < columnas:  # Derecha
                movimientos_posibles.append((x, y + 1))

            posicion_actual = random.choice(movimientos_posibles)
            camino.append(posicion_actual)
        
        poblacion.append(camino)
    
    return poblacion
    
def evaluate_fitness(self, individual: Individual, objective: Tuple[int, int] = (1, 1)):
        if individual.path[-1] == objective:
            individual.fitness = len(individual.path)
        else:
            distance = abs(individual.path[-1][0] - objective[0]) + abs(individual.path[-1][1] - objective[1])
            individual.fitness = len(individual.path) + (distance * self.penalty_factor)

def evaluate_population(self):
        for individual in self.population:
            self.evaluate_fitness(individual, (1, 1))

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


def select_tournament(self, tournament_size: int = 3) -> List[Individual]:
        selected = []
        for _ in range(self.population_size - self.elite_size):
            tournament = random.sample(self.population, tournament_size)
            winner = min(tournament, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected

def mutation(poblacion): 
    for path in poblacion:
        mutationpoint=random.randint(1,column-1)
        # print(f'mutationpoint is {mutationpoint}')
        path[mutationpoint]=(random.randint(1,row),path[mutationpoint][1])
    return poblacion

def display(lst):
    end=time.time()
    print(lst)
    a=agent(m,footprints=True)
    m.tracePath({a:lst},delay=10)
    # m.tracePath({b:m.path},delay=10)
    print(f'\n\nsolution occured in generation : {generation+1}')
    print(f'time taken to find solution is : {end-startt} seconds')
    m.run()
    plt.plot(best_in_pop)
    plt.xlabel('generations')
    plt.ylabel('min.fitness value per generation')
    plt.title(f'min.Fitness vs Generations using PopulationSize of {pop_size}\n(minimum fitness is best) \n')
    plt.show()


inicio = (1, 1)
meta = (10, 10)
poblacion = generar_poblacion(10, 10, 5, inicio, meta)
print(poblacion)

# Creacion del laberinto
m=maze(10,10)
m.CreateMaze(loopPercent=100)
m.run()

