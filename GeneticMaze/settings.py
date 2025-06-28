SQUARE_SIZE = 25 #Pixel size of the square image
N = 15 #NxN maze size (must be odd number)
FPS = 60 #frames per second
BACKGROUND_COLOR = (0,0,0)

START_POS = (1,0)
END_POS = (N-2, N-1)

GEN_COUNT = 100 #Max number of generations
TOP = 5 #Number of top agents to breed
ELITE = 5 #Number of elite agents to keep
POP = 100 #Total number of agents

BASES = ["U","D","L","R"]
DX = [0,0,-1,1]
DY = [-1,1,0,0]

DNA_LENGTH = int(2.5 * N * N)
MUTATION_CHANCE = 2#%