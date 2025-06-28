import random
import numpy as np
import settings

BASES = settings.BASES
DX = settings.DX
DY = settings.DY

DNA_LENGTH = settings.DNA_LENGTH
RUNNING_ID = 0
MUTATION_CHANCE = settings.MUTATION_CHANCE #%

#Calculates simple manhattan distance
def manhattan(pos1, pos2):
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    return dx + dy

class Agent:
    #Construct new genetic agent with generation and DNA sequence
    def __init__(self, generation, dna=None):
        global RUNNING_ID
        self.id = RUNNING_ID; RUNNING_ID += 1
        self.generation = generation
        if dna is not None:
            self.dna = dna #Preset DNA
        else:
            dna = np.empty(DNA_LENGTH, str) #Random DNA
            for i in range(DNA_LENGTH):
                dna[i] = random.choice(BASES)
            self.dna = dna
        
        self.pos = settings.START_POS #Begin at starting position
        self.index = 0 #Track what move agent is on

        self.fitness = 0 #Initialize to 0
        self.loops = 0 #Penalize for loops
        self.hits = 0 #Penalize for wall hits

    #Iterate to next position given DNA sequence
    def moveNext(self, maze):
        if self.index >= DNA_LENGTH: #Ran out of moves, DEATH
            return None, None
        
        i = BASES.index(self.dna[self.index])
        dx = DX[i]; dy = DY[i]
        old_pos = self.pos
        self.pos = (self.pos[0]+dy, self.pos[1]+dx)
        self.index += 1

        if (self.pos[0] < 0 or self.pos[0] >= settings.N or self.pos[1] < 0 or self.pos[1] >= settings.N or 
        maze[self.pos] == 1): #Hit wall or out of bounds, ignore
            self.pos = old_pos
            self.hits += 1
            return old_pos, old_pos
        
        #Check for loop (increase in manhattan distance)
        if manhattan(self.pos, settings.END_POS) > manhattan(old_pos, settings.END_POS):
            self.loops += 1

        #Update fitness (manhattan distance - wall hits - loops)
        dist = manhattan(self.pos, settings.END_POS)
        self.fitness = (2*settings.N - dist) - self.loops * 2 - self.hits

        #Check for reaching end (stop the agent)
        if dist == 0:
            self.fitness += 10000
            return None, None

        return old_pos, self.pos #Return old and new position (on grid)
        

    #Breed two agents with crossover and introduce some random mutation
    @staticmethod
    def breed(ag1, ag2):
        #Crossover DNA
        pivot = random.randint(0,DNA_LENGTH-1)
        dna = np.concatenate((ag1.dna[:pivot], ag2.dna[pivot:]))

        #Mutation, choose random bases (chance drops off with generation to keep genes more refined)
        chance = MUTATION_CHANCE
        for i in range(DNA_LENGTH):
            if random.randint(1,100) <= chance:
                dna[i] = random.choice(BASES) 
        
        return Agent(ag1.generation+1, dna)
    



    
