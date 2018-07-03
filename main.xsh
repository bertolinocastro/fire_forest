#!/usr/bin/xonsh


import numpy as np
import pymp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as matcolors
import matplotlib.patches as mpt
from collections import namedtuple

from sys import exit

# declaring a c-like stucture to hold the states
Enum = namedtuple('state', 'EMPASH SUSCEPTIBLE BURNING')

# defining enumeration just for code reading improvement
state = Enum(
    EMPASH = 0, # empty site or ashes of trees
    SUSCEPTIBLE = 1, # healthy tree
    BURNING = 2 # tree being burned
)

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
# pF = 0.03
pF = 0.00001

# grid dimension
dim = 500

# number of iterations
nt = 200

# number of threads to run
nThreads = 4

# grid definined in parallel mode
# grid using the fixed model for boundary conditions
grid = pymp.shared.array(shape=(2, dim+2, dim+2), dtype='uint8')

# setting the top left corner site to burning state in order to cheat the colormap function of matplotlib
# at the first step, it's setting all the healthy trees to Red (undesired behaviour)
grid[0][0][0] = state.BURNING
grid[1][0][0] = state.BURNING

# defining a default figure instance
fig = plt.figure(0)

# defining a color mapper (it will be mapped as a ratio with the high and low values inside the matrix)
colors = [
    '#e6e9ef', # mapping 0 to the gray color
    '#33b737', # mapping 1 to the green color
    '#e03131'  # mapping 2 to the red color
]

# defining a color mapper
cm = matcolors.ListedColormap(colors)

# defining a handle to the Patch instances for each legend element
handle = [mpt.Patch(color=colors[i]) for i in range(len(colors))]

# naming each legend element
legLabels = [
    'Empty or Ashes',
    'Healthy tree',
    'Burning tree'
]

# setting a informative title
plt.title('Forest fire model\ngrid dim %dx%d, pB=%.2f, pF=%.2f, nt=%d' %
    (dim,dim,pB,pF,nt))

# Put a legend below current axis
plt.legend(handles=handle, labels=legLabels, loc='upper center',
    bbox_to_anchor=(0.5, 0), fancybox=True, ncol=3)

# removing the axes labels and ticks
plt.gca().axis('off')

# defining variables to get the image output
images = []

with pymp.Parallel(nThreads) as p:
    # fig = plt.figure(p.thread_num)
    # variable to hold the actual plan in computation
    z = 0
    # time iteration
    for it in range(nt):
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

        if p.thread_num == 0:
            # plotting the actual state of the grid as a heatmap plot
            img = plt.imshow(grid[z], cmap=cm, animated=True)
            images.append([img])

# defining the fps value based on number of iterations (the gif must finish in 10 sec)
# or limiting it down to 25ms
# fps = nt/10
interval = max(10000/nt, 25)

# creating the animation with all states recorded and saving them as a mp4 file
an = anim.ArtistAnimation(fig, images, interval=interval, repeat_delay=1000)
an.save('drossel-schwabl_forest-fire-model_%d_%dx%d_pb%.2f_pf%.2f.mp4' %
    (nt, dim, dim, pB, pF), writer='ffmpeg', dpi=240)
