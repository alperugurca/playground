![Main](1.gif)

Source: https://www.youtube.com/watch?v=CGgdPvsbuJk  
Source: https://github.com/skylimitscience/GeneticMaze
# Genetic Maze
Python maze-solver that trains tiny agents to escape using DNA and evolution.
YouTube video: https://youtu.be/CGgdPvsbuJk

## Overview
The genetic algorithm employed in this project operates on three basic steps:
1. Simulation: agents carry out their DNA moveset within the physical maze
2. Selection: picking out top agents based on a fitness function
3. Reproduction: breeding agents to create new offspring using DNA crossover

### Simulation
Every genetic agent starts off with a DNA sequence that consists of directions (i.e. up, down, left, right). They follow their DNA instructions until the end of the sequence.

### Selection
Once agents have finished their route, we calculate a fitness score using a self-defined formula that takes into account distance, loopbacks, and wall collisions. A simple formula would be:
```python
- (dist + loopbacks + collisions)
```
or in the case of the actual code:
###### [agent.py](https://github.com/skylimitscience/GeneticMaze/blob/main/agent.py)
```python
self.fitness = (2*settings.N - dist) - self.loops * 2 - self.hits
```
We ultimately want less of all three, hence the negative sign. Given the fitness scores, we chose a certain number of agents (e.g. 5%) to randomly breed and also keep some for the next generation as "elite" members.

### Reproduction
Finally, to breed any two agents from our top pool, we use a single-point crossover of their DNA sequences by choosing a random pivot point and assigning the offspring with a DNA that merges the left half of parent 1 and right half of parent 2. Because maze-solving move sequences are order-sensitive, we don't want to choose a crossover method that breaks apart the sequence into too many pieces, hence just single-point. We also introduce mutations in DNA every once in a while (e.g 2% of the time), swapping out one direction for another, to promote genetic diversity (which is crucial in the evolutionary process).

###### [agent.py](https://github.com/skylimitscience/GeneticMaze/blob/main/agent.py)
```python
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
```
## Customization
If you're up for it, I highly encourage tinkering with the numbers in this project! You can change up most of the genetic conditions through the settings.py file and alter the fitness and breeding methods in the agent.py file, as outlined above.

###### [settings.py](https://github.com/skylimitscience/GeneticMaze/blob/main/settings.py)
```python
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
```
Happy coding!
