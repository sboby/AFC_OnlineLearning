
import genericTentacleModel
import particleTracker
import numpy as np


Nelements = 30
L = 0.15
r = 0.03

#Below values are arbitrary since prescribed motion
eModulus = 1
area = 1
density = 1
Izz = 1 

model = genericTentacleModel.PythonInterface \
(
    Nelements, 
    L, 
    r,
    eModulus,
    area,
    density, Izz
)

NTimeSteps = 2000
tArray = np.linspace ( 0, 10, num = NTimeSteps )
dt = tArray[1] - tArray[0]

densityArray = None
duration = 10 
# Domain parameters are defined within multiParticleTracker.py

for i in range ( NTimeSteps ):

    # VelocityArray = 1D-array of size Nx*Ny*2
    t = tArray[i]
    velocityArray = particleTracker.getVelocityArray ( particleTracker.X, particleTracker.Y, t )
    
    model.integrateTimeStep \
    ( 
        densityArray,
        velocityArray,
        dt 
    )

