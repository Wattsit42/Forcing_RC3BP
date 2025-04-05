import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import PIL
## IMPLEMENT STEP SIZE CONTROL FOR THE GAUSS METHOD

def calc_Lagrnage_Points(mu1,mu2):
    l1y = 0
    a1 = (mu2/(3*mu1))**(1/3)
    l1x = mu1 - a1 + a1**2/3 + a1**3/9 + 23*a1**4/81
    l2y = 0
    l2x = mu1 + a1 + a1**2/3 - a1**3/9 - 31*a1**4/81
    l3y = 0
    rat = mu2/mu1
    l3x = -mu2 - 1 + (7/12*rat)-7/12*(rat)**2 + 13223/20736*(rat)**3
    l4x = 1/2 - mu2
    l4y = np.sqrt(3)/2
    l5x = 1/2 - mu2
    l5y = -np.sqrt(3)/2
    return np.array([l1x,l2x,l3x,l4x,l5x]), np.array([l1y,l2y,l3y,l4y,l5y])

def apply_rotation_matrix(x,t):
    M = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    return M @ x

def derotate(x,times):
    derotated = np.array([])
    for i in range(len(x)):
        derotx = apply_rotation_matrix(x[i],times[i])
        derotated = np.append(derotated,derotx)
    derotated = np.reshape(derotated, (int(len(derotated)/2),2))
    return derotated

def restrictedH(t,x):
    # mu1 = 0.9990463 # Jupiter Sun system values
    # mu2 = 9.537e-4
    #
    mu1 = 0.98785 # Earth Moon system values
    mu2 = 1.215e-2
    r1 = np.sqrt((x[2] + mu2)**2 + x[3]**2)
    r2 = np.sqrt((x[2] - mu1)**2 + x[3]**2)
    q1dot = x[0] + x[3]
    q2dot = x[1] - x[2]
    pxdot = -x[2] + x[1] - mu1 * (x[2] + mu2)/r1**3 - mu2*(x[2] - mu1)/r2**3 + x[2] + x_direction_force_term(t)
    pydot = -x[3] - x[0] - mu1 * x[3]/r1**3 - mu2 * x[3]/r2**3 + x[3] + y_direction_force_term(t)
    return np.array([pxdot,pydot,q1dot,q2dot])
# Indicator function attempt 1, leads to some bad behaviour as these functions are not Lipschitz continuous, but this on/off behaviour is really what we want.
# def x_direction_force_term(t):
#     if t <= 0.03:
#         return -30
#     elif 12.5 <= t <= 12.6:
#         return -4
#     elif 48<= t <= 48.01:
#         return 5
#     else:
#         return 0
#
# def y_direction_force_term(t):
#     if t <= 0.06:
#         return -30
#     elif 12.5 <= t <= 12.8:
#         return -2
#     elif 48 <= t <= 48.01:
#         return 0
#     else:
#         return 0

def gaussian(magnitude, shift, stretch, t):
    return magnitude * np.exp(-stretch*(t-shift)**2)

## We want to give gaussian functions a try, but I am skeptical. Could be quite hard to control.
def x_direction_force_term(t):
    return gaussian(-1,6,5,t)

def y_direction_force_term(t):
    return gaussian(-18.7,0.2,795,t) + gaussian(1,5.5,8,t) + gaussian(-3.6,15,900,t) + gaussian(-30,16,900,t)

class ButcherTab:
    def __init__(self,A,b,c):
        self.A = A
        self.b = b
        self.c = c

class Gauss:
    #While this is named after the Gauss method, it would really work for an arbitrary Runge-Kutta method
    def __init__(self, func, t_0, x_0, h, max_t, Butcher):
        self.func = func
        self.t_0 = t_0
        self.x_0 = x_0
        self.max = max_t
        self.h = h
        self.B = Butcher

    def integrate(self,maxit, eps=1e-14):
        xsarr = np.array([self.x_0])
        times = np.array(self.t_0)
        #first_integrals = np.array(calculate_first_integrals(self.x_0))
        xn = self.x_0
        t_n = self.t_0
        while t_n+self.h <= self.max:
            xn1 = self.step(xn, t_n, maxit,eps)
            xsarr = np.append(xsarr, xn1)
            t_n += self.h
            times = np.append(times, t_n)
            xn = xn1
            #first_integrals = np.append(first_integrals, calculate_first_integrals(xn))
        return xsarr, times#, first_integrals

    def step(self, xn, t_n, maxit, eps):
        x_array = np.full((len(self.B.c),len(xn)),xn) # Populates initial guess for stages
        for m in range(maxit):
            z_array = x_array.copy() # copies to store values and compare to check convergence
            for i in range(len(self.B.c)): #computes stages
                ki = self.func(t_n + self.h*self.B.c[i],self.interior_staging(i,xn,z_array)) # calculate new stage
                x_array[i] = ki.copy() # add it to the old array
            if np.sum(abs(x_array-z_array)) <= eps: # if the difference in the two is below the tolerance then we accept these values
                return xn + self.h * self.sum_of_stages(x_array) # calculate the value at the next time step
        raise

    def interior_staging(self, i, xn, ks):
        stage_sum = np.zeros(len(xn))
        for j in range(len(self.B.A[i])):
            stage_sum += self.B.A[i][j] * ks[j]
        return xn + self.h * stage_sum

    def sum_of_stages(self,stages):
        stage_sum = np.zeros(len(stages[-1]))
        for i in range(len(stages)):
            stage_sum += stages[i]*self.B.b[i]
        return stage_sum

#lx,ly = calc_Lagrnage_Points(0.9990463,9.537e-4) #Sun Jupiter Lagrange Point approximations
lx, ly = calc_Lagrnage_Points(0.98785,1.215e-2) #Earth Moon Lagrange Point approximations

#x0 = np.array([0.0,0.99,0.99,0.0])
#x0 = np.array([0.0,1.1,0.6,0.0])
#x0 = np.array([0.0,1.1,0.5,0.0]) # First Gaussian Earth Moon example
x0 = np.array([0.2,3.4,0.1,0.0])

t0=0
h = 0.01
max_t = 20

GO6 = ButcherTab([[5/36,2/9 - (np.sqrt(15))/15,5/36 - (np.sqrt(15))/30],[5/36+(np.sqrt(15))/24,2/9,5/36-np.sqrt(15)/24],[5/36+np.sqrt(15)/30,2/9+np.sqrt(15)/15,5/36]],[5/18,4/9,5/18],[1/2-np.sqrt(15)/10,1/2,1/2+np.sqrt(15)/10])
Gl = Gauss(restrictedH, t0, x0, h, max_t, GO6)
xn,times = Gl.integrate(20)
xn = np.reshape(xn, (int(len(xn)/4),4))
# This gets us the intertial frame coordinates
rotation_coords = np.column_stack((xn[:,2],xn[:,3]))
derotated_coords = derotate(rotation_coords,times)
# Sets up the figure
fig, ax = plt.subplots(1,1)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)

# m1coords = np.array([-9.537e-4,0]) # Sun-Jupiter
# m2coords = np.array([0.9990463,0])

m1coords = np.array([-1.215e-2,0]) # Earth-Moon
m2coords = np.array([0.98785,0])
#Rotating Frame plot
# plt.scatter(m1coords[0],m1coords[1], label='m1')
# plt.scatter(m2coords[0],m2coords[1],label = 'm2')
# plt.plot(xn[:,2],xn[:,3], label='Rotating Frame Orbit')
# plt.scatter(xn[-1,2],xn[-1,3], label='Satellite')

m1inertial_frame = apply_rotation_matrix(m1coords,max_t)
m2inertial_frame = apply_rotation_matrix(m2coords,max_t)
# Inertial frame plot
# plt.scatter(m1inertial_frame[0],m1inertial_frame[1], label = 'm1') #Gives us the positions of the planets at the end of
# plt.scatter(m2inertial_frame[0],m2inertial_frame[1], label='m2') #the simulation
# plt.plot(derotated_coords[:,0],derotated_coords[:,1],label='Non-Rotating Orbit')
# plt.scatter(derotated_coords[-1,0],derotated_coords[-1,1],label='Satellite')


# Animation Section

orbit_line = ax.plot(derotated_coords[0,0],derotated_coords[0,1],label='Non-Rotating Orbit')[0]
m1_plot = ax.scatter(m1coords[0],m1coords[1], label = 'm1')
m2_plot = ax.scatter(m2coords[0],m2coords[1], label = 'm2')
m3_plot = ax.scatter(x0[2],x0[3],label = 'm3')
def update(frame):
    # Update the orbit line
    x = derotated_coords[:frame,0]
    y = derotated_coords[:frame,1]
    orbit_line.set_xdata(x)
    orbit_line.set_ydata(y)
    # Update the scatter plots
    m1_pos = apply_rotation_matrix(m1coords,times[frame])
    m2_pos = apply_rotation_matrix(m2coords,times[frame])
    m3_pos = derotated_coords[frame]
    m1_plot.set_offsets(m1_pos)
    m2_plot.set_offsets(m2_pos)
    m3_plot.set_offsets(m3_pos)
    return orbit_line,m1_plot,m2_plot,m3_plot
ani = FuncAnimation(fig, update, frames=len(times), interval=1)



# plt.xlim(0.9,1.1)
# plt.ylim(-0.1,0.1)
plt.legend(loc='upper right')
plt.title("t~" + str(max_t/(2*np.pi))[:5] + " Months")
ani.save(filename="pillow_example.gif", writer="pillow")
plt.show()