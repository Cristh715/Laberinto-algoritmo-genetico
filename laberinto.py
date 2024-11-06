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