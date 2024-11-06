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

inicio = (1, 1)
meta = (10, 10)
poblacion = generar_poblacion(10, 10, 5, inicio, meta)
print(poblacion)

# Creacion del laberinto
m=maze(10,10)
m.CreateMaze(loopPercent=100)
m.run()
