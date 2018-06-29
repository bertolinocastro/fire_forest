#!/usr/bin/xonsh


import numpy as np
import pymp
from enum import Enum

# defining enumeration just for code reading improvement
class state(Enum):
    EMPASH = 0 # empty site or ashes of trees
    SUSCEPTIBLE = 1 # healthy tree
    BURNING = 2 # tree being burned

# defining a function to check wether the actual tree has burning friends
def hasCloserFire(grid,pos):
    x, y, z = pos
    return  grid[z][x+1][y] == state.BURNING or \
            grid[z][x-1][y] == state.BURNING or \
            grid[z][x][y+1] == state.BURNING or \
            grid[z][x][y-1] == state.BURNING

# probability of birth
pB = 0.3

# probability of ignite
pF = 0.3

# grid dimension
dim = 20

# grid definined in parallel mode
# grid using the fixed model for boundary conditions
# dt = np.dtype('O')
# grid = pymp.shared.array(shape=(2, dim+2, dim+2), dtype=dt)
grid = pymp.shared.list()
# creating dimensions and initializing grid values
grid.append([]); grid.append([])
grid[0] = [[0 for x in range(dim+2)] for y in range(dim+2)]
grid[1] = [[0 for x in range(dim+2)] for y in range(dim+2)]

with pymp.Parallel(4) as p:
    # variable to hold the actual plan in computation
    z = 0
    # time iteration
    for it in range(100):
        # spatial iteraions
        for x in p.xrange(1, 1+dim):
            for y in range(1, 1+dim):
                newCell = cell = grid[z][x][y]
                if cell == state.BURNING: # if it's already burning, turn it into ashes
                    newCell = state.EMPASH
                elif cell == state.EMPASH: # if it's empty or an ashe
                    if np.random.rand() <= pB: # if the god wants a tree to be born, make a tree
                        newCell = state.SUSCEPTIBLE
                else: # if it's a healthy tree
                    # if its closer friends are in fire or the god wants it to get burned, turn into a burning tree
                    if hasCloserFire(grid,(x,y,z)) or np.random.rand() <= pF:
                        newCell = state.BURNING

                # updating the new value to the next plan
                grid[1-z][x][y] = newCell

        # switching to next plan
        z = 1 - z
