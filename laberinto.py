from pyamaze import maze,agent,COLOR
# Creacion del laberinto
m=maze(20,20)
m.CreateMaze(loopPercent=100)
m.run()