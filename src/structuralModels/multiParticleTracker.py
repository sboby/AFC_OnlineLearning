import numpy as np 
import matplotlib.pyplot as plt 
import math as m 
import pdb 

"======PARTICLE TRACKING TRIAL 3============"



### Domain parameters

Lx = 2.5
Ly = 0.41 
fn = 5
Nx = 251
Ny = 42
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)

xIndices = np.linspace(0, Nx-1, Nx)
yIndices = np.linspace(0, Ny-1, Ny)

xCoordinates = np.linspace(0, Lx, Nx)
yCoordinates = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(xCoordinates, yCoordinates)

measurementPointX = (Lx - Lx/3)
### Particle class

class Particle ():
    def __init__ ( self ):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.passedBoundary = False

    def setPosition ( self, x, y ):
        self.position[0] = x 
        self.position[1] = y 

    def setVelocity ( self, vx, vy ):
        self.velocity[0] = vx 
        self.velocity[1] = vy 


### Let's make particles 


def yInlet ( t ):
    frequency = 0.245
    return ( Ly * np.sin(frequency*t*2*np.pi) + Ly ) / 2 


def weight ( deltaTime, T, tau ):
    if deltaTime < tau:
        return 0.5 * ( 1 - np.cos ( np.pi/tau* deltaTime ) )
    elif deltaTime > T - tau:
        return 0.5 * ( 1 + np.cos ( np.pi/tau * ( deltaTime - ( T - tau ) ) ) )
    else:
        return 1
    


### Initialisation 
class ParticleSet ():
    def __init__ ( self, tWindow, tau ):
        self.particleList = []
        self.timestepCounter = 0
        self.passedUpper = 0 
        self.passedLower = 0 

        self.passedUpperTimestamps = []
        self.passedLowerTimestamps = [] 
        self.tWindow = tWindow 
        self.tau = tau
        self.tlist = []
        self.rtlist = []
        


    def updateParticleList(self, xInitial, yInitial, t):
        self.timestepCounter += 1
        
        if self.timestepCounter % 1 ==0: 
            particleY = yInlet ( t )
            particle = Particle ()
            particle.setPosition ( xInitial, particleY )
            self.particleList.append ( particle )



    ### updateParticlePositions
    def updateParticlePositions ( self, velocityArray, t, dt ):
        nParticles = len ( self.particleList )
        ax.cla()

        outOfBounds = []
        boundaryPassed = []

        for i in range(nParticles):
            particle = self.particleList[i]

            particlePosition = particle.position
            particleX = particlePosition[0]
            particleY = particlePosition[1]


            ### Optimize later - round py_new to nearest integer (normalise with Ly) and use indecing
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
            
            particleXNew = particleX + particleVelocityX * dt
            particleYNew = particleY + particleVelocityY * dt

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

            ax.plot(particleXNew, particleYNew, 'go')


        for particle_idx in outOfBounds[::-1]: 
            self.particleList.pop(particle_idx)
        


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
        # plt.legend()
        plt.pause(0.001)

        # print ( self.passedLower, self.passedUpper )


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

duration    = 10
dt          = 0.01

nTimeSteps  = int(duration/dt)
t           = 0

fig, ax = plt.subplots(figsize = (12, 12))


tWindow = 1
tau = 0.25
particleSet = ParticleSet ( tWindow, tau )

while t<duration:     
    velocityArray = getVelocityArray ( X, Y, t )

    particleSet.updateParticlePositions ( velocityArray, t, dt )

    xInitial, yInitial =  ( 0, Ly/3 )

    particleSet.updateParticleList(xInitial, yInitial, t)
    particleSet.computeParticleConcentration ( t )

    t += dt

tlist, rtlist = particleSet.finaliseSimulation ()

plt.show()

plt.plot (tlist, rtlist)
plt.show()