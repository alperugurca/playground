import pygame
import math
import time
import random
import numpy as np
from agent import Agent
import settings

pygame.init()

SS = settings.SQUARE_SIZE #Pixel size of the square image
N = settings.N #NxN maze size
FPS = settings.FPS #frames per second

WIDTH, HEIGHT = SS * N, SS * N
BACKGROUND_COLOR = settings.BACKGROUND_COLOR
PI = math.pi

screen = pygame.display.set_mode((WIDTH, HEIGHT))
square_img = pygame.image.load("square.png")
"""
agent_img = pygame.image.load("agent.png")
var = pygame.PixelArray(agent_img)
var.replace((255,255,255),(255,0,0)); del var
"""

clock = pygame.time.Clock()
running = True

maze = np.ones((N, N))
START_POS = settings.START_POS
END_POS = settings.END_POS

#Helps achieve smooth gliding movement
def sq_tween(a, b, t):
    return a + (b - a) * t

def sq_tween_pos(a, b, t):
    return (sq_tween(a[0], b[0], t), sq_tween(a[1], b[1], t))

#Clamp a number to a range
def clamp(x, a, b):
  return max(min(x, b), a)


"""
=========================================================
                    MAZE GENERATION
=========================================================
"""

#Visualize the maze
def visualize_maze():
    global maze, SS, N
    screen.fill(BACKGROUND_COLOR)
    for i in range(N):
        for j in range(N):
            if maze[i][j] == 0:
                screen.blit(square_img, (j * SS, i * SS))

#Uses recursive backtracking to generate maze
vis = np.zeros((N, N))
dx = [1,0,-1,0]
dy = [0,1,0,-1]
def gen_aux(i, j):
    global maze, vis, N
    vis[i][j] = 1

    order = [0,1,2,3]
    random.shuffle(order)
    #Visit random neighbors and knock out wall
    for k in order:
        x = i+dy[k]*2; y = j+dx[k]*2
        if x<1 or x>=N or y<1 or y>=N: continue #Check for borders
        if vis[x][y] != 1:
            maze[x-dy[k]][y-dx[k]] = 0
            gen_aux(x,y)
    

def generate_maze():
    global maze, vis, N

    #Fill in borders and form pockets
    for i in range(1,N-1):
        for j in range(1,N-1):
            if i%2 == 1 and j%2 == 1:
                maze[i][j] = 0
    
    #Generate inner maze
    vis = np.zeros((N, N))
    gen_aux(1,1)

    #Put start and end
    maze[START_POS] = 0; maze[END_POS] = 0

"""
=========================================================
                    GENETIC AGENTS
=========================================================
"""

#Visualize movement of agents given their start and end positions
#agents: {agent_id -> [start_pos, end_pos, fitness] **on grid}, ensure that they are still alive/valid
def visualize_agents(agents):
    for frame in range(FPS+1):
        t = frame/60
        visualize_maze()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False

        for id in agents:
            fitness = agents[id][2]
            pos = sq_tween_pos(agents[id][0], agents[id][1], t)
            pygame.draw.circle(screen, (0,0,255), (pos[0] + SS/2, pos[1] + SS/2), 5)
        
            #screen.blit(agent_img, pos)
            
        pygame.display.update()
        time.sleep(1/FPS/100)
    

#Handle simulation from visualizing, tracking, and creating a subsequent generation
def simulate():
    alive = []
    for i in range(settings.POP):
        ag = Agent(0)
        alive.append(ag)
    
    for gen in range(settings.GEN_COUNT):
        dead = []
        vis_agents = {}
        #Simulate current generation
        while len(alive) > 0:
            #print("-> Alive:", len(alive), "|", "Generation:",gen)
            for i in range(len(alive)-1,-1,-1):
                start_pos, end_pos = alive[i].moveNext(maze)
                id = alive[i].id
                if start_pos is None: #Dead
                    dead.append(alive[i])
                    alive.pop(i)
                    if id in vis_agents:
                        del vis_agents[id]
                else: #Move on display
                    #Convert positions to pixel display
                    i1, j1 = start_pos
                    i2, j2 = end_pos
                    start_pos = (j1 * SS, i1 * SS)
                    end_pos = (clamp(j2 * SS + random.randint(-5,5), (j2-0.5)*SS, (j2+0.5)*SS),
                               clamp(i2 * SS + random.randint(-5,5), (i2-0.5)*SS, (i2+0.5)*SS))
                        #*^ Add a bit of randomness to end position (visual effect only), so they don't stack
        
                    start_vis = start_pos
                    if id in vis_agents:
                        start_vis = vis_agents[id][1] #Display start position is previous end position
                    vis_agents[id] = [start_vis, end_pos, alive[i].fitness]
            if len(alive) > 0 and (gen in [0,1,2,10,20,50,99]): #Change list to which generations you want to visualize
                visualize_agents(vis_agents)
        
        fitness_tot = 0
        for ag in dead: fitness_tot += ag.fitness
        print("Generation:",gen,"|","Avg. Fitness:",round(fitness_tot/len(dead),2))

        
        #Find top performers and breed
        top = sorted(dead, key=lambda x: x.fitness, reverse=True)[:settings.TOP]
        for i in range(settings.ELITE): #Elitism, keep some individuals
            alive.append(Agent(top[i].generation+1, top[i].dna))

        for i in range(settings.POP - settings.ELITE): #Breed new offspring
            alive.append(Agent.breed(random.choice(top), random.choice(top)))

generate_maze()
#visualize_maze()
simulate()

    
    
    
