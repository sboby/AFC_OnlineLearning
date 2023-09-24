from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import math as m 
import sys
import pdb 
import particleTracker as particleTracker
from io import BytesIO
from PIL import Image

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

sys.path.append('..')

class ComplexTentacle(): 
    def __init__(self, N_elements, L, r, dt, duration ): 
        self.N = N_elements
        self.L = L 
        self.t = 0 
        self.s = np.linspace(0,1, self.N+1)

        dt = 0.00115547
        
        
        # tentacle shape parameters 
        self.r = r 
        self.duration = duration 
        self.dt = dt 

        surfaceLength   = np.sqrt(self.r**2 + self.L**2)
        self.ds         = L/(self.N)
        self.dsSurface  = surfaceLength/(self.N+1)
        self.timeSteps = int(duration/self.dt)
        self.timeStep = 0         


        "----------------Adding the flush cylinder --------------------------"
        # this is the distance from the center of the 
        # circle to the base of the tenticle 
        self.A = self.r**2/self.L
        self.r_c = np.sqrt(self.A**2 + self.r**2)
        self.theta_0 = np.arcsin(self.r / self.r_c)
                # # number of nodes in the arc
        self.N_c = int((2*np.pi - 2*self.theta_0)*self.r_c/self.dsSurface)
        self.N_c = 48

        dtheta  = 2 * (np.pi - self.theta_0) / self.N_c
        self.xc = np.linspace(0,0, num=self.N_c)
        self.yc = np.linspace(-self.r_c, self.r_c*np.cos(self.theta_0) , self.N_c)
        self.theta = np.linspace ( self.theta_0, 2*np.pi-self.theta_0 - dtheta, num = self.N_c)
        
        
        self.N_total = (self.N)*2 + (self.N_c)

      
        self.x_ca   = np.linspace( 0, 0, num=self.N+1, dtype=float)  
        self.y_ca   = np.linspace( self.A, (self.L + self.A), num=self.N+1, dtype=float)  

        self.theta_0c = np.pi/2-self.theta_0

        "================INITIAL POSITION ARRAYS================="
        self.xc_0  = np.zeros(self.N_c)
        self.yc_0  = np.zeros(self.N_c)
        self.xp_u0 = np.zeros(self.N)
        self.yp_u0 = np.zeros(self.N)
        self.xp_l0 = np.zeros(self.N)
        self.yp_l0 = np.zeros(self.N)
        self.alphaPrevious = np.zeros(self.N+1)

        self.initializeTentacle()

        "=================OUTPUT ARRAYS==========================="
        self.displacements = np.zeros(self.N_total*6)
        self.velocities = np.zeros(self.N_total*6)


        "====================DATA TO EXPORT========================"
        self.q1_history = []
        self.q2_history = [] 


    def initializeTentacle(self): 
        '''
        specifies the resting start position of the tentacle 
        to be used for displacement calculations 
        '''

        x_upper_extended = np.zeros((self.N+1))
        y_upper_extended = np.zeros((self.N+1))
        x_lower_extended = np.zeros((self.N+1))
        y_lower_extended = np.zeros((self.N+1))

        for i in range(0,self.N_c):   
            x = self.r_c*np.sin(self.theta[i])
            y = self.r_c*np.cos(self.theta[i])
            
            self.xc_0[i] = x 
            self.yc_0[i] = y 


        for i in range(0, self.N+1):
            s_c = self.s[i]
            alpha = self.q1(self.t)*s_c + (self.q2(self.t)*s_c**2)/2

            self.alphaPrevious[i] = alpha 

            x_p_u, y_p_u, x_p_l, y_p_l = self.calulatePerimeter( self.x_ca[i], self.y_ca[i], s_c, alpha)
            x_upper_extended[i] = x_p_u
            y_upper_extended[i] = y_p_u

            x_lower_extended[i] = x_p_l
            y_lower_extended[i] = y_p_l
        
        self.xp_u0 = x_upper_extended[:-1]
        self.yp_u0 = y_upper_extended[:-1]
        self.xp_l0 = x_lower_extended[1:]
        self.yp_l0 = y_lower_extended[1:]


    
    def updateLoading(self, loading): 
        self.loading = loading.copy()

    def q2(self, t): 
       offset_time = 10 
       max_time = 12
       max_time_adj = max_time - offset_time

       if t< offset_time: 
          return 0 

       elif t> max_time:
          return m.pi
       
       t_adj = t - offset_time
       
       return (m.pi)*m.sin(m.pi*t_adj/(2*max_time_adj))

        
    def q1(self, t): 
       if t>-1 : 
           return 0 
       else: 
          return 0
        
    
    def alpha0s(self,t): 
        return m.sin(1*t) * 0

    def angular_velocity(self, t):
        return  (self.alpha0s(t) - self.alpha0s(t-self.dt))/self.dt 

    def calulatePerimeter(self, x_c, y_c, s_c, alpha): 
        
        x_p_u = x_c - self.r*(1-s_c)*m.cos(alpha)
        y_p_u = y_c - self.r*(1-s_c)*m.sin(alpha)
        
        x_p_l = x_c + self.r*(1-s_c)*m.cos(alpha)
        y_p_l = y_c + self.r*(1-s_c)*m.sin(alpha)

        return(x_p_u, y_p_u, x_p_l, y_p_l)


    def calculateAxialVelocity(self, i, velocities_x, velocities_y, alpha, alpha_prev): 
        v_x = velocities_x[i-1] - m.cos(alpha)*self.ds*((alpha-alpha_prev)/self.dt)
        v_y = velocities_y[i-1] - m.sin(alpha)*self.ds*((alpha-alpha_prev)/self.dt)

        return v_x, v_y

    def calculatePerimeterVelocity(self, v_x, v_y, s_c, alpha, alpha_prev): 
        vx_p_u = v_x - self.r*(1 - s_c)*(-np.sin(alpha))*((alpha-alpha_prev)/self.dt)
        vy_p_u = v_y - self.r*(1 - s_c)*(np.cos(alpha))*((alpha-alpha_prev)/self.dt)

        vx_p_l = v_x + self.r*(1 - s_c)*(-np.sin(alpha))*((alpha-alpha_prev)/self.dt)
        vy_p_l = v_y + self.r*(1 - s_c)*(np.cos(alpha))*((alpha-alpha_prev)/self.dt)
        
        return vx_p_u, vy_p_u, vx_p_l, vy_p_l
        
    
    def integrateTimeStep ( self, dt, plot = False, ax = None): 
        xp_u_array_full = np.zeros(self.N + 1)
        xp_l_array_full = np.zeros(self.N + 1)
        yp_u_array_full = np.zeros(self.N + 1)
        yp_l_array_full = np.zeros(self.N + 1)
        xp_u_array      = np.zeros(self.N)
        xp_l_array      = np.zeros(self.N)
        yp_u_array      = np.zeros(self.N)
        yp_l_array      = np.zeros(self.N)

        
        velocities_x = np.zeros(self.N+1, dtype=float)
        velocities_y = np.zeros(self.N+1, dtype=float)
        
        # per time step perimeter velocities 
        velocities_xp_u_full  = np.zeros(self.N + 1, dtype=float)
        velocities_yp_u_full  = np.zeros(self.N + 1, dtype=float)
        velocities_xp_l_full  = np.zeros(self.N + 1, dtype=float)
        velocities_yp_l_full  = np.zeros(self.N + 1, dtype=float)
        velocities_xp_u       = np.zeros(self.N, dtype=float)
        velocities_yp_u       = np.zeros(self.N, dtype=float)
        velocities_xp_l       = np.zeros(self.N, dtype=float)
        velocities_yp_l       = np.zeros(self.N, dtype=float)


        alphaCurrent = np.zeros(self.N+1)

        alpha0 = self.alpha0s(self.t + dt)

        omega_c = self.angular_velocity(self.t + dt)
        
        ### Circle stuff 
        for i in range(0,self.N_c):
            
            x = self.r_c*np.sin(self.theta[i])
            y = self.r_c*np.cos(self.theta[i])
            
            self.xc[i] = x 
            self.yc[i] = y 

        ### The definition of alpha0 is inconsistent with alpha, I think
        self.x_ca[0] = self.A * np.sin(alpha0 * 0)
        self.y_ca[0] = self.A * np.cos(alpha0 * 0)

        ### Update the central axis
        q1 = self.q1(self.t+dt) 
        q2 = self.q2(self.t+dt)
        self.q1_history.append(q1)
        self.q2_history.append(q2)

        for i in range(1, self.N+1): 
            s_c         = self.s[i]
            alpha       = q1*s_c + q2*s_c**2/2
            
            alpha_prev  = self.alphaPrevious[i]

            alphaCurrent[i] = alpha

            self.x_ca[i] = self.x_ca[i-1] - m.sin(alpha)*self.ds
            self.y_ca[i] = self.y_ca[i-1] + m.cos(alpha)*self.ds


            v_x, v_y = self.calculateAxialVelocity(i, velocities_x, velocities_y, alpha, alpha_prev)
            velocities_x[i] = v_x
            velocities_y[i] = v_y


        ### Update the perimeter surface
        for i in range(0, self.N + 1): 

            s_c         = self.s[i]
            alpha       = q1*s_c + q2*s_c**2/2
            
            alpha_prev  = self.alphaPrevious[i]
            
            # -----------------------------------------------
            # central axis coordinates 
            x_ca = self.x_ca[i]
            y_ca = self.y_ca[i]
            vx_ca = velocities_x[i]
            vy_ca = velocities_y[i]
            
            # -----------------------------------------------
            # # perimeter coordinates 
            x_p_u, y_p_u, x_p_l, y_p_l = self.calulatePerimeter( x_ca, y_ca, s_c, alpha)
            xp_u_array_full[i] = x_p_u
            yp_u_array_full[i] = y_p_u
            xp_l_array_full[i] = x_p_l
            yp_l_array_full[i] = y_p_l

            
            # ----------------VELOCITY CALCULATIONS---------------------------------------

            # axial velocities  
            # perimeter velocities 
            vx_p_u, vy_p_u, vx_p_l, vy_p_l = self.calculatePerimeterVelocity(vx_ca, vy_ca, s_c, alpha, alpha_prev)
            
            # ---------- per time step the velocity and displacement for the entire tentacle stored 
            # as seperate arrays 
            velocities_xp_u_full[i] = vx_p_u
            velocities_yp_u_full[i] = vy_p_u
            velocities_xp_l_full[i] = vx_p_l
            velocities_yp_l_full[i] = vy_p_l

            # numerical velocity calculation             
            
            #----------------------------------------------------

        xp_u_array = xp_u_array_full[:-1]
        yp_u_array = yp_u_array_full[:-1]
        xp_l_array = xp_l_array_full[1:]
        yp_l_array = yp_l_array_full[1:]

        velocities_xp_u = velocities_xp_u_full[:-1]
        velocities_yp_u = velocities_yp_u_full[:-1]
        velocities_xp_l = velocities_xp_l_full[1:]
        velocities_yp_l = velocities_yp_l_full[1:]

        "======================================================================================="
        

        self.alphaPrevious = alphaCurrent


        "-------ROTATING COORDINATES FOR NON-ZERO ALPHAS-------------------------------"
        x_c_r = self.yc*np.sin(alpha0) + self.xc*np.cos(alpha0)
        y_c_r = self.yc*np.cos(alpha0) - self.xc*np.sin(alpha0)
        
        XpArray_ur = yp_u_array*np.sin(alpha0) + xp_u_array*np.cos(alpha0)
        YpArray_ur = yp_u_array*np.cos(alpha0) - xp_u_array*np.sin(alpha0)
                
        XpArray_lr = yp_l_array*np.sin(alpha0) + xp_l_array*np.cos(alpha0)
        YpArray_lr = yp_l_array*np.cos(alpha0) - xp_l_array*np.sin(alpha0)

        "-----------ROTATING VELOCITY VECTORS FOR NON-ZERO ALPHA0S----------------------"
    
        velocities_xp_ur = velocities_yp_u*np.sin(alpha0) + velocities_xp_u*np.cos(alpha0)
        velocities_yp_ur = velocities_yp_u*np.cos(alpha0) - velocities_xp_u*np.sin(alpha0)

        velocities_xp_lr = velocities_yp_l*np.sin(alpha0) + velocities_xp_l*np.cos(alpha0)
        velocities_yp_lr = velocities_yp_l*np.cos(alpha0) - velocities_xp_l*np.sin(alpha0)

        "----------ANGULAR VELOCITY CALCULATION------------------------"
        # circle perimeter 
        vy_c = -x_c_r*omega_c
        vx_c =  y_c_r*omega_c

        # tenticle perimeter             
        vxt_pu = velocities_xp_ur + YpArray_ur*omega_c
        vyt_pu = velocities_yp_ur - XpArray_ur*omega_c

        vxt_pl = velocities_xp_lr + YpArray_lr*omega_c
        vyt_pl = velocities_yp_lr - XpArray_lr*omega_c

        "------------------------recording displacements--------------------------------------"
        dx_c = x_c_r - self.xc_0
        dx_l = XpArray_lr - self.xp_l0
        dx_u = XpArray_ur - self.xp_u0
        
        dy_c = y_c_r - self.yc_0
        dy_l = YpArray_lr - self.yp_l0
        dy_u = YpArray_ur - self.yp_u0
        
        '----------FSI inputs---------------------------------------------------------------------'
        #------ displacement in x 
        dx                              = np.zeros(self.N_total)
        dx[:self.N]                     = dy_u
        dx[self.N:2*self.N]             = dy_l[::-1]
        dx[2*self.N:]                   = dy_c
        
        #------ displacement in y 
        dy                              = np.zeros(self.N_total)
        dy[:self.N]                     = -dx_u
        dy[self.N:2*self.N]             = -dx_l[::-1]
        dy[2*self.N:]                   = -dx_c
                
        #------- velocity x 
        vx_p                            = np.zeros(self.N_total)
        vx_p[:self.N]                   = vyt_pu
        vx_p[self.N:2*self.N]           = vyt_pl[::-1]
        vx_p[2*self.N:]                 = vy_c
        
        #------- velocity y 
        vy_p                            = np.zeros(self.N_total)
        vy_p[:self.N]                   = -vxt_pu
        vy_p[self.N:2*self.N]           = -vxt_pl[::-1]
        vy_p[2*self.N:]                 = -vx_c

        displacements = np.zeros(self.N_total*6)
        displacements[::6]  = dx
        displacements[1::6] = dy

        velocities = np.zeros(self.N_total*6)
        velocities[::6]   = vx_p
        velocities[1::6]  = vy_p

        
        '-----------------------------------------------------------------------------------------------'

        if plot == True:
      
          ax.plot(YpArray_ur+ 0.5, -XpArray_ur+0.2, c = color_palette()[0])
          ax.plot(YpArray_lr+ 0.5, -XpArray_lr+0.2, c = color_palette()[0])
          
          ax.plot(y_c_r + 0.5, -x_c_r+0.2, c = color_palette()[0])
        

        output = np.zeros(2*self.N_total*6)
        output[:self.N_total*6] = displacements
        output[self.N_total*6:] = velocities

   
        self.t += dt

        return output
    
        
### Change values here! 
tWindow                     = 1
tau                         = 0.25
timeStepSkipUpdate          = 1
timeStepSkipAddition        = 50 
timeStepSkipPlot            = 10
timeEnd                     = 50
Lx                          = 2.5
Nx                          = 501
tentacle_xc, tentacle_yc    = 0.2, 0.2 

def yInlet ( t ):
    frequency = 0.245
    Ly = 0.41

    return ( Ly/2 * np.sin(frequency*t*2*np.pi) + Ly ) / 2 

class PythonInterface ():
  def __init__ ( self, Nelements, L, r, eModulus, area, density, Izz  ):
    self.tentacle     = ComplexTentacle( Nelements, L, r, 0.1, 0)
    self.particleSet  = particleTracker.ParticleSet ( tWindow, tau, timeStepSkipUpdate, timeStepSkipAddition, timeEnd, yInlet )
    self.t            = 0
    self.timeIndex    = 0
    self.imList       = []
    self.fig, self.axs = plt.subplots(2)
   


  def updateLoading ( self, loading ):
    self.tentacle.updateLoading ( loading )
    
  def integrateTimeStep ( self, densityArray, velocityArray, dt  ):

    self.particleSet.collectVelocityMeasurements(self.tentacle.r_c, velocityArray, tentacle_xc, tentacle_yc, self.t, dt)


    self.particleSet.updateTimeStep ( velocityArray, self.t, dt )
    self.particleSet.computeParticleConcentration ( self.t )

  

    self.t += dt

    if self.timeIndex % timeStepSkipPlot == 0:
      self.axs[0].clear()
      self.axs[1].clear()


      particlePlotAx      = self.axs[0]
      concentrationPlotAx = self.axs[1]

      self.particleSet.plotParticles ( particlePlotAx )
      self.particleSet.plotVorticityArray ( velocityArray, self.t, particlePlotAx )
      self.particleSet.plotConcentration ( concentrationPlotAx )
      output = self.tentacle.integrateTimeStep ( dt, True, particlePlotAx )


      particlePlotAx.set_xlim(0, 2.5)
      particlePlotAx.set_ylim(0, 0.41)
      particlePlotAx.set_aspect('equal')
      concentrationPlotAx.set_aspect ( 5 )
      particlePlotAx.set_title ( r"$t = $ " + str(self.t)[:5] + r" $[s]$" )
      concentrationPlotAx.set_title ( r"Concentration as function of time" )
      concentrationPlotAx.set_xlabel ( r"$t$ [$s$]" )
      concentrationPlotAx.set_ylabel ( r"$p$ [-]" )

      buf = BytesIO()
      plt.savefig(buf, dpi = 200, format = None) 
      buf.seek(0)
      im = Image.open(buf)
      self.imList.append(im)

      plt.pause(0.01)

    else:       
      output = self.tentacle.integrateTimeStep ( dt )

    if self.t > timeEnd:
      periodFrame = dt * timeStepSkipPlot
      print ( periodFrame )

      fps = 1 / periodFrame

      print ( fps )
      fpsFactor = fps / 50
      frameSkip = m.ceil ( fpsFactor )

      #=-----------DATASET CREATION  -----------
      
      exportData = np.zeros((self.timeIndex+1, 3))
      concentrationHistory = np.array(self.particleSet.rtlist)
      q1_history = np.array(self.tentacle.q1_history)
      q2_history = np.array(self.tentacle.q2_history)
      velocitySensorMeasurements = np.array(self.particleSet.velocityMeasurements)
      print(velocitySensorMeasurements.shape)
      
      print("q1_history.shape", q1_history.shape)
      print("self.timeIndex", self.timeIndex)
      exportData[:, 0] = q1_history
      
      exportData[:, 1] = q2_history
      
      exportData[:, 2] = concentrationHistory


      velocityData = np.zeros((self.timeIndex+1, 3, 2))
      velocityData[:, :, :] = velocitySensorMeasurements
      
      print(velocitySensorMeasurements)

      np.savetxt('stepResponse_2.24.txt', exportData, header = "q1, q2, concentration")
   

      self.imList[0].save('stepResponse_2.24.gif', loop = 0, save_all=True, append_images=self.imList[frameSkip::frameSkip], fps = fps/frameSkip, dpi = 200, duration = frameSkip*1000/fps )
      quit()


    self.timeIndex += 1
    return output
    

  def finaliseTimeStep ( self, dt ):
    pass