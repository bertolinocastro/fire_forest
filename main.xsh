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

# --------------------
# fixed declarations
# --------------------

# number of threads to run
nThreads = 1

# declaring a c-like stucture to hold the states
Enum = namedtuple('state', 'EMPASH SUSCEPTIBLE BURNING')

# defining enumeration just for code reading improvement
state = Enum(
    EMPASH = 0, # empty site or ashes of trees
    SUSCEPTIBLE = 1, # healthy tree
    BURNING = 2 # tree being burned
)

# defining a color mapper (it will be mapped as a ratio with the high and low values inside the matrix)
colors = [
    '#e6e9ef', # mapping 0 to the gray color
    '#33b737', # mapping 1 to the green color
    '#e03131'  # mapping 2 to the red color
]

# defining a color mapper
cm = matcolors.ListedColormap(colors)


# --------------------
# func definitions
# --------------------

# defining a function to check wether the actual tree has burning friends
def hasCloserFire(grid,pos):
    x, y, z = pos
    return  grid[z][x+1][y] == state.BURNING or \
            grid[z][x-1][y] == state.BURNING or \
            grid[z][x][y+1] == state.BURNING or \
            grid[z][x][y-1] == state.BURNING

def attCell(grid,x,y,z,pB,pF):
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

def runModel(dim, pB, pF, nt, save=False, monteCarlo=False):
    # creating output dir
    mkdir -p @('out/dim=%04d_pb=%.7f_pf=%.7f'%(dim,pB,pF))

    # grid definined in parallel mode
    # grid using the fixed model for boundary conditions
    grid = pymp.shared.array(shape=(2, dim+2, dim+2), dtype='uint8')

    # setting the top left corner site to burning state in order to cheat the colormap function of matplotlib
    # at the first step, it's setting all the healthy trees to Red (undesired behaviour)
    grid[0][0][0] = state.BURNING
    grid[1][0][0] = state.BURNING

    bioM = np.zeros(len(nt))
    fireM = np.zeros(len(nt))

    if monteCarlo:
        sync = pymp.shared.array(shape=(nThreads))
        for i in range(len(sync)):
            sync[i] = 0

    with pymp.Parallel(nThreads) as p:

        if monteCarlo:
            # computing parallel vars
            length = int(dim/nThreads)
            remain = dim%nThreads
            lb = p.thread_num*length + 1
            ub = (1+p.thread_num)*length + 1
            if p.thread_num == nThreads-1 : ub += remain

        # variable to hold the actual plan in computation
        z = 0

        # time iteration
        for it in nt:
            mkdir -p @('out/dim=%04d_pb=%.7f_pf=%.7f'%(dim,pB,pF))
            if monteCarlo:
                sync[p.thread_num] = it
                # print(p.thread_num, 'it ', it)
                # defining a point of syncronuization betweens its
                while int(np.sum(sync)/nThreads) < it:
                    # print(p.thread_num, 'espera por ', sync)
                    pass
                # print(p.thread_num,"saiu")

                posMCx = np.random.randint(1,dim+1,(ub-lb)*dim) # x dim
                posMCy = np.random.randint(lb,ub,(ub-lb)*dim) # y dim

                # spatial iteraions
                for x,y in zip(posMCx,posMCy):
                    attCell(grid,x,y,z,pB,pF)
            else:
                # spatial iteraions
                for x in p.range(1, 1+dim):
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

            # computing tree's biomass left
            if p.thread_num == 0:
                bioM[it] = np.sum(grid[z,1:dim+1,1:dim+1] == state.SUSCEPTIBLE)/dim**2
                fireM[it] = np.sum(grid[z,1:dim+1,1:dim+1] == state.BURNING)/dim**2

            if p.thread_num == 0 and save:
                # plotting the actual state of the grid as a heatmap plot
                plt.figure(1)
                img = plt.imshow(grid[z], cmap=cm)
                # img = plt.imsave('out/dim=%04d_pb=%.7f_pf=%.7f/state_%06d.png'%(dim,pB,pF,it), grid[z], cmap=cm)
                # plt.matshow(grid[z], cmap=cm)
                plt.axis("off")
                plt.savefig('out/dim=%04d_pb=%.7f_pf=%.7f/state_%06d.png'%(dim,pB,pF,it))
                plt.close()
                # fifi.show()
                # images.append([img])

    # defining the fps value based on number of iterations (the gif must finish in 10 sec)
    # or limiting it down to 25ms
    # fps = nt/10
    # interval = max(10000/nt, 25)

    # creating the animation with all states recorded and saving them as a mp4 file
    # an = anim.ArtistAnimation(fig, images, interval=interval, repeat_delay=1000)
    # an.save('drossel-schwabl_forest-fire-model_%d_%dx%d_pb%.2f_pf%.2f.mp4' %
    #     (nt, dim, dim, pB, pF), writer='ffmpeg', dpi=240)

    # creating a mp4 video from images
    if save:
        cd @('out/dim=%04d_pb=%.7f_pf=%.7f'%(dim,pB,pF))
        # mogrify -resize 320x240 *.png
        ffmpeg -framerate 24 -i state_%06d.png -c:v libx264 -pix_fmt yuv420p -r 30 -y out.mp4
        cd -


    # sending back useful information
    return {'bioMass':bioM,'fireMass':fireM}


def plotOut(x,y,path,title,lbx,lby,limx=0,limy=1,scatter=False):
    # plotting biomass over pF
    fi = plt.figure(0);
    fi.clf()
    ax = fi.gca()
    ax.set_title(title)
    ax.set_ylabel(lby); ax.set_xlabel(lbx)
    if scatter:
        ax.scatter(x,y,color='k')
    else:
        ax.plot(x,y,'.-k')
    ax.set_ylim((0,1))
    fi.savefig(path)
# ------------------------------------------------------------


# probability of birth
_pB = 1.

# probability of ignite
# pF = 0.03
_pF = np.linspace(0,1,20)

# grid dimension
_dim = np.arange(6,52,9)

# number of iterations
_nt = np.arange(200)

mkdir -p bioMass dimm

bioMasses = np.zeros(len(_pF))

fireCrit = np.zeros(len(_dim))

# looping over dimension
for j, idim in enumerate(_dim):
    print("idim looping : %04d"%idim)
    # looping over fire probability
    for i, ipF in enumerate(_pF):
        if(idim != 24 and ipF != _pF[-3]):
            continue
        print("ipF looping : %.7f"%ipF)


        toSave = True if idim == _dim[0] and ipF == _pF[1] else False
        toSave=True

        # running the model for the desired parameters
        res = runModel(idim, _pB, ipF, _nt,save=toSave)

        bioMasses[i] = res['bioMass'][-1]

        plotOut(
            _nt,res['bioMass'],
            'bioMass/bioMassVsTimestep_dim=%04d_pb=%.7f_pf=%.7f.png'%(idim,_pB,ipF),
            'Biomass Vs. Timestep\ndim=%04d pB=%.7f pF=%.7f'%(idim, _pB, ipF),
            'it','# trees'
        )

    # plotting biomass over pF
    plotOut(
        _pF, bioMasses,
        'bioMass/biomassVspF_dim=%04d_pb=%.7f.png'%(idim,_pB),
        'Biomass Vs. Fire probability\ndim=%04d pB=%.7f'%(idim, _pB),
        'pF','# trees'
    )

    # getting where the system has vanished, if any
    x = np.where(bioMasses == 0)[0]
    fireCrit[j] = x[0] if len(x) > 0 else np.nan

plotOut(
    _dim,fireCrit,
    'dimm/fireCritVsdim_pb=%.7f.png'%_pB,
    'Critical pF Vs. Grid Dim\tpB=%.7f'%_pB,
    'dim','$pF_c$',
    scatter=True
)
