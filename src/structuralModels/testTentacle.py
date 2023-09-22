from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import math as m 
import sys
import pdb 
import src.struct.particleTracker as particleTracker
from io import BytesIO
from PIL import Image

sys.path.append('..')


""" -----------------------------------------------------------------
INPUT: 
L = 
N_elements = 
r = 

OUTPUT: 
displacements = [dx, dy, dz, theta_x, theta_y, theta_z]
velocities = [vx, vy, vz, theta_x_dot, theta_y_dot, theta_z_dot]

[displacements, velocities ]
per time-step 
-----------------------------------------------------------------""" 

class ComplexTentacle(): 
    def __init__(self, N_elements, L, r, dt, duration ): 
        self.N = N_elements
        self.L = L 
        self.t = 0 
        self.s = np.linspace(0,1, self.N+1)
        
        
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
        
        '------trying to fix the element problem--------'
        # # number of nodes in the arc
        self.N_c = int((2*np.pi - 2*self.theta_0)*self.r_c/self.dsSurface)
        self.N_c = 48

        
        '-------------'
        dtheta  = 2 * (np.pi - self.theta_0) / self.N_c
        self.xc = np.linspace(0,0, num=self.N_c)
        self.yc = np.linspace(-self.r_c, self.r_c*np.cos(self.theta_0) , self.N_c)
        self.theta = np.linspace ( self.theta_0, 2*np.pi-self.theta_0 - dtheta, num = self.N_c)
        
        "----------------------------------------------------------"

        self.N_total = (self.N)*2 + (self.N_c)

        # initial position of the tentacle 
        # defined to start the motion completely horizontal 
        # self.x0     = np.linspace(0, 0, num=self.N, dtype=float)
        # self.y0     = np.linspace(self.A, (self.L + self.A), num = self.N, dtype=float)
        self.x_ca   = np.linspace( 0, 0, num=self.N+1, dtype=float)   # XArray 
        self.y_ca   = np.linspace( self.A, (self.L + self.A), num=self.N+1, dtype=float)  # YArray 
        
        # self.x[0] = self.x0[0]
        # self.y[0] = self.y0[0]

        ''''
        trying shit 
        '''
        self.theta_0c = np.pi/2-self.theta_0
        '---------------'

        "================INITIAL POSITION ARRAYS==================="
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


        "========================================================="

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

        # for i in range(0,self.N):
        #     s_c = self.s[i]
        #     alpha = self.q1(s_c,self.t)*s_c + (self.q2(s_c,self.t)*s_c**2)/2

        #     # -----------------------------------------------
        #     # central axis coordinates 
        #     self.x_ca[i] = self.x_ca[i-1] - sin(alpha)*self.ds
        #     self.y_ca[i] = self.y_ca[i-1] + cos(alpha)*self.ds

        #     # self.x0[i] = x_ca
        #     # self.y0[i] = y_ca
            
        #     # -----------------------------------------------
        #     # # perimeter coordinates 

        # print("HERE")

        for i in range(0, self.N+1):
            s_c = self.s[i]
            alpha = self.q1(s_c,self.t)*s_c + (self.q2(s_c,self.t)*s_c**2)/2

            x_p_u, y_p_u, x_p_l, y_p_l = self.calulatePerimeter( self.x_ca[i], self.y_ca[i], s_c, alpha)
            x_upper_extended[i] = x_p_u
            y_upper_extended[i] = y_p_u

            x_lower_extended[i] = x_p_l
            y_lower_extended[i] = y_p_l
        
        self.xp_u0 = x_upper_extended[:-1]
        self.yp_u0 = y_upper_extended[:-1]
        self.xp_l0 = x_lower_extended[1:]
        self.yp_l0 = y_lower_extended[1:]

        # self.xp_u0[0] = -self.r + self.dsSurface*np.sin(self.theta_0c) # upper corner of the tenatcle base 
        # self.yp_u0[0] = self.A + self.dsSurface*np.cos(self.theta_0c) 
        # self.xp_l0[0] = self.r  - self.dsSurface*np.sin(self.theta_0c) # lower corner of the tentacle base 
        # self.yp_l0[0] = self.A  + self.dsSurface*np.cos(self.theta_0c)

        # self.xp_u0[-1] = 0 - self.ds*np.sin(self.theta_0c) # upper corner of the tenatcle base 
        # self.yp_u[-1] = self.L + self.A - self.ds*np.cos(self.theta_0c) 
        
        # print(self.xp_u0, self.yp_u0)
        # print(self.xp_l0, self.yp_l0)
        # print(self.xc_0, self.yc_0)
        
        # plt.title('Hello I am now a Ten Tickle')
        # plt.rcParams.update({'font.size': 14})
        # plt.xlim(-0.5, 0.5)
        # plt.ylim(-0.5, 0.5)
        # plt.plot(self.yp_u0, self.xp_u0, 'b')
        # plt.plot(self.yp_l0, self.xp_l0, 'b')
        # plt.plot(self.yc_0, self.xc_0, 'g')
        # plt.grid()
        # plt.gca().set_aspect('equal')
        # plt.show()
            

    
    def updateLoading(self, loading): 
        self.loading = loading.copy()

    def q1(self, s,t): 
        return m.sin(4*t)
        
    def q2(self, s,t): 
        return m.sin(4*t) 
    
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

    

    def plotAnimation(self):
        # TODO: this doesn't work yet 
        plt.cla()
        plt.title('Time Dependent Tentacle: Iteration 01')
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)
        plt.plot(self.x,self.y)
        plt.plot(self.xp_u, self.yp_u, 'r')
        plt.plot(self.xp_l, self.yp_l, 'b')
        plt.pause(0.1)
        

    def velocityPlots(self, plotTimeStep):
        # TODO: finsh implementing velocity against displacement plots 

        # plt.plot(velocity_y_full_manual[:, plotTimeStep]*dt, 'r', label='numerical velocity')
        # plt.plot(velocity_y_full_analytic[:,plotTimeStep]*dt, 'b',  label="analytic velocity")
        # plt.plot(dy_complete[:, plotTimeStep], 'g', label="displacement t=1")
        # plt.legend() 
        # plt.show()

        pass
    
            
    
    def integrateTimeStep ( self, dt, plot = False, ax = None): 
        xp_u_array_full = np.zeros(self.N + 1)
        xp_l_array_full = np.zeros(self.N + 1)
        yp_u_array_full = np.zeros(self.N + 1)
        yp_l_array_full = np.zeros(self.N + 1)
        xp_u_array      = np.zeros(self.N)
        xp_l_array      = np.zeros(self.N)
        yp_u_array      = np.zeros(self.N)
        yp_l_array      = np.zeros(self.N)

        "-------------------------------------------------------------------------------------------"
    
        # manual_velocities_x  = np.zeros(self.N, dtype=float)
        # manual_velocities_y = np.zeros(self.N, dtype=float)
        
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

        "---------------------------------------------------------"

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
        for i in range(1, self.N+1): 
            s_c         = self.s[i]
            alpha       = self.q1(s_c,self.t+dt)*s_c + (self.q2(s_c,self.t+dt)*s_c**2)/2
            alpha_prev  = self.alphaPrevious[i]

            alphaCurrent[i] = alpha

            self.x_ca[i] = self.x_ca[i-1] - m.sin(alpha)*self.ds
            self.y_ca[i] = self.y_ca[i-1] + m.cos(alpha)*self.ds

            # print(self.y_ca[i], self.x_ca[i])

            # x_old = prev_x[i,self.timeStep-1]
            # y_old = prev_y[i,self.timeStep-1]
            # prev = (x_old, y_old)

            v_x, v_y = self.calculateAxialVelocity(i, velocities_x, velocities_y, alpha, alpha_prev)
            velocities_x[i] = v_x
            velocities_y[i] = v_y

            # manual_velocities_x[i] = (x_ca - prev[0])/self.dt
            # manual_velocities_y[i] = (y_ca - prev[1])/self.dt

        ### Update the perimeter surface
        for i in range(0, self.N + 1): 

            s_c         = self.s[i]
            alpha       = self.q1(s_c,self.t+dt)*s_c + (self.q2(s_c,self.t+dt)*s_c**2)/2
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
        
        # velocity_x_full_manual[:,self.timeStep] = manual_velocities_x
        # velocity_x_full_analytic[:,self.timeStep] = velocities_x

        # velocity_y_full_manual[:,self.timeStep] = manual_velocities_y
        # velocity_y_full_analytic[:,self.timeStep] = velocities_y


        # perimter velocities 
        # perimeter_vx_u[:,self.timeStep] = velocities_xp_u
        # perimeter_vy_u[:,self.timeStep] = velocities_yp_u

        # perimeter_vx_l[:,self.timeStep] = velocities_xp_l
        # perimeter_vy_l[:,self.timeStep] = velocities_yp_l


        self.alphaPrevious = alphaCurrent
        # self.prev_x[:,self.timeStep] = self.x
        # self.prev_y[:,self.timeStep] = self.y

        "-------ROTATING COORDINATES FOR NON-ZERO ALPHAS-------------------------------"
        x_c_r = self.yc*np.sin(alpha0) + self.xc*np.cos(alpha0)
        y_c_r = self.yc*np.cos(alpha0) - self.xc*np.sin(alpha0)
        
        # XArray_r = self.y*np.sin(alpha0) + self.x*np.cos(alpha0)
        # YArray_r = self.y*np.cos(alpha0) - self.x*np.sin(alpha0)

        XpArray_ur = yp_u_array*np.sin(alpha0) + xp_u_array*np.cos(alpha0)
        YpArray_ur = yp_u_array*np.cos(alpha0) - xp_u_array*np.sin(alpha0)
                
        XpArray_lr = yp_l_array*np.sin(alpha0) + xp_l_array*np.cos(alpha0)
        YpArray_lr = yp_l_array*np.cos(alpha0) - xp_l_array*np.sin(alpha0)

        "-----------ROTATING VELOCITY VECTORS FOR NON-ZERO ALPHA0S----------------------"
        # velocities_xr = velocities_y*np.sin(alpha0) + velocities_x*np.cos(alpha0)
        # velocities_yr = velocities_y*np.cos(alpha0) - velocities_x*np.sin(alpha0)

        velocities_xp_ur = velocities_yp_u*np.sin(alpha0) + velocities_xp_u*np.cos(alpha0)
        velocities_yp_ur = velocities_yp_u*np.cos(alpha0) - velocities_xp_u*np.sin(alpha0)

        velocities_xp_lr = velocities_yp_l*np.sin(alpha0) + velocities_xp_l*np.cos(alpha0)
        velocities_yp_lr = velocities_yp_l*np.cos(alpha0) - velocities_xp_l*np.sin(alpha0)

        "----------ANGULAR VELOCITY CALCULATION------------------------"
        # circle perimeter 
        vy_c = -x_c_r*omega_c
        vx_c =  y_c_r*omega_c

        # TODO: these still don't work ask sam 
        # tenticle axial 
        # vx_axial = velocities_xr + YArray_r*omega_c
        # vy_axial = velocities_yr - XArray_r*omega_

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

        # pdb.set_trace()
        '-----------------------------------------------------------------------------------------------'

        if plot == True:
        # plt.title('Time Dependent Tentacle: Iteration 01')
        # plt.rcParams.update({'font.size': 14})
        # plt.xlim(-0.5, 0.5)
        # plt.ylim(-0.5, 0.5)
        # plt.plot(self.yp_u0, -self.xp_u0, 'r')
        # plt.plot(self.yp_l0, -self.xp_l0, 'r')
          ax.plot(YpArray_ur+ 0.5, -XpArray_ur+0.2, 'b')
          ax.plot(YpArray_lr+ 0.5, -XpArray_lr+0.2, 'b')
          # plt.plot(self.y_ca, -self.x_ca, 'k')
          ax.plot(y_c_r + 0.5, -x_c_r+0.2, 'g')
        # plt.grid()

        output = np.zeros(2*self.N_total*6)
        output[:self.N_total*6] = displacements
        output[self.N_total*6:] = velocities

        # print(np.shape(output))
        # quit(1)

        self.t += dt

        return output
    

tentacle = ComplexTentacle ( 30, 0.15, 0.06, 0.1, 0 )

