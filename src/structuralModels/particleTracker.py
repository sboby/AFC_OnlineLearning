import numpy as np 
import matplotlib.pyplot as plt 
import math as m 
import pdb 
import sys

sys.path.append('..')

from matplotlib import rc
import matplotlib as mpl
import seaborn as sns
import matplotlib.font_manager
from seaborn.palettes import color_palette
import matplotlib.ticker as ticker

sns.set_style("dark")
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 
mpl.rcParams['figure.dpi'] = 200


### Domain parameters -- update manually!

Lx = 2.5
Ly = 0.41 
fn = 5
Nx = 501
Ny = 83
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)

xIndices = np.linspace(0, Nx-1, Nx)
yIndices = np.linspace(0, Ny-1, Ny)

xCoordinates = np.linspace(0, Lx, Nx)
yCoordinates = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(xCoordinates, yCoordinates)

measurementPointX = 1.25

class Particle ():
    def __init__ ( self ):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.passedBoundary = False
        self.color = 'k'

    def setPosition ( self, x, y ):
        self.position[0] = x 
        self.position[1] = y 

    def setVelocity ( self, vx, vy ):
        self.velocity[0] = vx 
        self.velocity[1] = vy 

    def setColor ( self, color ):
        self.color = color


### Let's make particles 


def yInlet ( t ):
    frequency = 0.245
    return Ly/2.1
    # return ( Ly * np.sin(frequency*t*2*np.pi) + Ly ) / 2 


def weight ( deltaTime, T, tau ):
    if deltaTime < tau:
        return 0.5 * ( 1 - np.cos ( np.pi/tau* deltaTime ) )
    elif deltaTime > T - tau:
        return 0.5 * ( 1 + np.cos ( np.pi/tau * ( deltaTime - ( T - tau ) ) ) )
    else:
        return 1
    

class ParticleSet ():
    def __init__ ( self, tWindow, tau, timeStepSkipUpdate, timeStepSkipAddition, timeEnd, initialYFunction ):
        self.timeStepSkipUpdate   = timeStepSkipUpdate
        self.timeStepSkipAddition = timeStepSkipAddition
        self.initialYFunction     = initialYFunction
        self.tWindow              = tWindow 
        self.timeEnd              = timeEnd
        self.tau                  = tau

        self.particleList     = []
        self.timestepCounter  = 0
        self.passedUpper      = 0 
        self.passedLower      = 0 

        self.passedUpperTimestamps  = []
        self.passedLowerTimestamps  = [] 
        self.tlist                  = []
        self.rtlist                 = []


        # position of sensor measurements 
        self.velocitySensorPlacement = [] 
        self.velocityMeasurements    = []
    
    #------------------- NEW FEATURE ---------------------------------
    def collectVelocityMeasurements(self, r, velocityArray, tentacle_xc, tentacle_yc, t, dt):
        # 1. Define Sensor Positions 
        # SENSOR 1 : LEADING EDGE 
        xs_0, ys_0 = (tentacle_xc - r), tentacle_yc
        # SENSOR 2 : UPPER EDGE 
        xs_1, ys_1 = tentacle_xc, (tentacle_yc + r)
        # SENSOR 3 : LOWER EDGE 
        xs_2, ys_2 = tentacle_xc, (tentacle_yc - r)

        self.velocitySensorPlacement = [(xs_0, ys_0), (xs_1, ys_1), (xs_2, ys_2)]

        # 2. Find velocities at sensor locations based on the 
        # return velocity array 
        velocityArray = np.reshape ( velocityArray, (Nx, Ny, 2 ))

        velocityMeasurements = np.zeros((len(self.velocitySensorPlacement), 2))

        for idx, sensor in enumerate(self.velocitySensorPlacement):
            sensorX = sensor[0]
            sensorY = sensor[1]

            xInt    = sensorX/dx
            yInt    = sensorY/dx 

            x1 = int ( xInt )
            x2 = x1 + 1
            y1 = int ( yInt )
            y2 = y1 + 1
            
            v1_x, v1_y = velocityArray[x1, y1,:]
            v2_x, v2_y = velocityArray[x1, y2,:]
            v3_x, v3_y = velocityArray[x2, y1,:]
            v4_x, v4_y = velocityArray[x2, y2,:]

            sensorVelocityX = (v1_x*(x2-xInt)*(y2-yInt) + 
                v3_x*(xInt - x1)*(y2-yInt) + 
                v2_x*(x2 - xInt)*(yInt - y1)+ 
                v4_x*(xInt-x1)*(yInt - y1))/((x2-x1)*(y2-y1))
            sensorVelocityY = (v1_y*(x2-xInt)*(y2-yInt) + 
                v3_y*(xInt - x1)*(y2-yInt) + 
                v2_y*(x2 - xInt)*(yInt - y1)+ 
                v4_y*(xInt-x1)*(yInt - y1))/((x2-x1)*(y2-y1))
            
            velocityMeasurements[idx][0]  = sensorVelocityX
            velocityMeasurements[idx][1]  = sensorVelocityY
    
        self.velocityMeasurements.append(velocityMeasurements)

  

    def updateTimeStep ( self, velocityArray, t, dt ):
        if self.timestepCounter % self.timeStepSkipAddition == 0:
            self.updateParticleList ( t )

        if self.timestepCounter % self.timeStepSkipUpdate == 0:
            self.updateParticlePositions ( velocityArray, t, dt )

        self.timestepCounter += 1

    def updateParticleList(self, t):

        
        particleY = self.initialYFunction ( t )
        particle = Particle ()
        particle.setPosition ( 0, particleY )
        
        if particleY > Ly / 2:
            particle.setColor ( color_palette()[1] )
        else:
            particle.setColor ( color_palette()[2] )

        self.particleList.append ( particle )



    def updateParticlePositions ( self, velocityArray, t, dt ):
        nParticles = len ( self.particleList )

        outOfBounds = []
        boundaryPassed = []

        velocityArray = np.reshape ( velocityArray, (Nx, Ny, 2 ))

        for i in range(nParticles):
            particle = self.particleList[i]

            particlePosition = particle.position
            particleX = particlePosition[0]
            particleY = particlePosition[1]

            xInt = particleX/dx        
            yInt = particleY/dx

            x1 = int ( xInt )
            x2 = x1 + 1
            y1 = int ( yInt )
            y2 = y1 + 1
            
            v1_x, v1_y = velocityArray[x1, y1,:]
            v2_x, v2_y = velocityArray[x1, y2,:]
            v3_x, v3_y = velocityArray[x2, y1,:]
            v4_x, v4_y = velocityArray[x2, y2,:]

            particleVelocityX = (v1_x*(x2-xInt)*(y2-yInt) + 
                v3_x*(xInt - x1)*(y2-yInt) + 
                v2_x*(x2 - xInt)*(yInt - y1)+ 
                v4_x*(xInt-x1)*(yInt - y1))/((x2-x1)*(y2-y1))
            particleVelocityY = (v1_y*(x2-xInt)*(y2-yInt) + 
                v3_y*(xInt - x1)*(y2-yInt) + 
                v2_y*(x2 - xInt)*(yInt - y1)+ 
                v4_y*(xInt-x1)*(yInt - y1))/((x2-x1)*(y2-y1))
            
            particle.setVelocity ( particleVelocityX, particleVelocityY )
            
            particleXNew = particleX + particleVelocityX * dt * self.timeStepSkipUpdate
            particleYNew = particleY + particleVelocityY * dt * self.timeStepSkipUpdate

            if particleXNew > Lx: 
                outOfBounds.append(i)

            if particleXNew > measurementPointX and particle.passedBoundary == False: 
                self.particleList[i].passedBoundary = True

                if particleY > Ly/2 : 
                    self.passedUpper += 1
                    self.passedUpperTimestamps.append(t)

                else : 
                    self.passedLower += 1
                    self.passedLowerTimestamps.append(t)

            particle.setPosition ( particleXNew, particleYNew )



        for particle_idx in outOfBounds[::-1]: 
            self.particleList.pop(particle_idx)
        
  

    def plotParticles ( self, ax ):
        nParticles = len ( self.particleList )

        for i in range(nParticles):
            particle = self.particleList[i]

            markerColor = particle.color

            ax.plot(particle.position[0], particle.position[1], c = markerColor, marker = 'o', markersize = 2 )

    def plotVorticityArray ( self, velocityArray, t, ax ):
        velocityArray = np.reshape ( velocityArray, (Nx, Ny, 2 ))

        dx = Lx / Nx
        vorticityArray =  ( velocityArray[1:-1,2:,0] - velocityArray[1:-1,:-2,0] ) / ( 2 * dx )
        vorticityArray -= ( velocityArray[2:,1:-1,1] - velocityArray[:-2,1:-1,1] ) / ( 2 * dx )


        fQuiver = 4
        ax.imshow  \
            ( 
            vorticityArray.transpose(), 
            alpha = 0.5,
            origin = 'lower',
            extent=[0, Lx, 0, Ly],
            vmin = -75,
            vmax = 75,
            cmap = 'bwr',
            )
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')
        ax.axhline ( 0 )
        ax.plot ( [1., 1.75], [Ly/2, Ly/2], c = color_palette()[0] )
        ax.axhline ( Ly )
        ax.axvline ( measurementPointX, ls = '--', c = color_palette()[7])

    def plotQuivers ( self, velocityArray, t ):
        ax = plt.gca()
        velocityArray = np.reshape ( velocityArray, (Nx, Ny, 2 ))

        fQuiver = 4
        ax.quiver \
            (X[::fQuiver,::fQuiver], 
            Y[::fQuiver,::fQuiver], 
            velocityArray[::fQuiver,::fQuiver,0].transpose(), 
            velocityArray[::fQuiver,::fQuiver,1].transpose(), 
            alpha = 0.1)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')
        plt.axhline ( 0 )
        plt.axhline ( Ly/2 )
        plt.axhline ( Ly )
        plt.axvline ( measurementPointX )
        plt.title ( str(t) )

    def plotConcentration ( self, ax ):
        print ( "WHY" )
        ax.plot ( self.tlist, self.rtlist )
        ax.set_xlim ( 0, self.timeEnd )
        ax.set_ylim ( 0, 1 )
        ax.grid ( which = 'major')

    def computeParticleConcentration(self, t): 
        
        deleteLowerList = []
        for i, time in enumerate ( self.passedLowerTimestamps ):
            if time < t - self.tWindow:
                deleteLowerList.append(i)

        for index in deleteLowerList[::-1]:
            self.passedLowerTimestamps.pop ( index )
        
        deleteUpperList = []
        for i, time in enumerate ( self.passedUpperTimestamps ):
            if time < t - self.tWindow:
                deleteUpperList.append(i)

        for index in deleteUpperList[::-1]:
            self.passedUpperTimestamps.pop ( index )
        
        wUpperSum = 0
        wLowerSum = 0

        for timeStamp in self.passedUpperTimestamps: 
            wUpperSum += weight ( t-timeStamp, self.tWindow, self.tau ) 


        for timeStamp in self.passedLowerTimestamps: 
            wLowerSum += weight ( t-timeStamp, self.tWindow, self.tau ) 

        if wUpperSum + wLowerSum == 0:
            rt = 0
        else:
            rt = wLowerSum/(wUpperSum + wLowerSum)
        
        print ( rt, len(self.passedLowerTimestamps), len(self.passedUpperTimestamps) )

        self.rtlist.append ( rt )
        self.tlist.append ( t )

    def finaliseSimulation ( self ):
        return self.tlist, self.rtlist

def getVelocityArray ( X, Y, t ):
    xShape = np.shape(X)
    result = np.zeros((xShape[1], xShape[0], 2))

    result[:,:,0] = np.abs(np.sin(2*np.pi*Y/Ly)).transpose()
    result[:,:,1] = (0.1*np.abs(np.sin(2*np.pi*Y/Ly))**2).transpose()*np.sin(fn*2*np.pi*t)

    return result
