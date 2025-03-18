import numpy as np
import matplotlib.pyplot as plt
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
    M =np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    return M @ x

def derotate(x,times):
    derotated = np.array([])
    for i in range(len(x)):
        derotx = apply_rotation_matrix(x[i],times[i])
        derotated = np.append(derotated,derotx)
    derotated = np.reshape(derotated, (int(len(derotated)/2),2))
    return derotated

def restrictedH(t,x):
    mu1 = 0.9990463
    mu2 = 9.537e-4
    r1 = np.sqrt((x[2] + mu2)**2 + x[3]**2)
    r2 = np.sqrt((x[2] - mu1)**2 + x[3]**2)
    q1dot = x[0] + x[3]
    q2dot = x[1] - x[2]
    pxdot = -x[2] + x[1] - mu1 * (x[2] + mu2)/r1**3 - mu2*(x[2] - mu1)/r2**3 + x[2] + x_direction_force_term(t)
    pydot = -x[3] - x[0] - mu1 * x[3]/r1**3 - mu2 * x[3]/r2**3 + x[3] + y_direction_force_term(t)
    return np.array([pxdot,pydot,q1dot,q2dot])

def x_direction_force_term(t):
    if t <= 0.03:
        return -30
    elif 12.5 <= t <= 12.6:
        return -4
    elif 48<= t <= 48.01:
        return 5
    else:
        return 0

def y_direction_force_term(t):
    if t <= 0.06:
        return -30
    elif 12.5 <= t <= 12.8:
        return -2
    elif 48 <= t <= 48.01:
        return 0
    else:
        return 0

class ButcherTab:
    def __init__(self,A,b,c):
        self.A = A
        self.b = b
        self.c = c

class Gauss:
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

lx,ly = calc_Lagrnage_Points(0.9990463,9.537e-4)

#x0 = np.array([-ly[4],lx[4]+0.01,lx[4]+0.01,ly[4]])
x0 = np.array([0.0,0.99,0.99,0.0])
#x0 = np.array([-0.01,lx[2],lx[2],0.01])

t0=0
h = 0.001
max_t = 70

GO6 = ButcherTab([[5/36,2/9 - (np.sqrt(15))/15,5/36 - (np.sqrt(15))/30],[5/36+(np.sqrt(15))/24,2/9,5/36-np.sqrt(15)/24],[5/36+np.sqrt(15)/30,2/9+np.sqrt(15)/15,5/36]],[5/18,4/9,5/18],[1/2-np.sqrt(15)/10,1/2,1/2+np.sqrt(15)])
Gl = Gauss(restrictedH, t0, x0, h, max_t, GO6)
xn,times = Gl.integrate(20)
xn = np.reshape(xn, (int(len(xn)/4),4))

plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)

plt.scatter(0.9990463,0)
#plt.scatter(xn[-1,2],xn[-1,3])
plt.scatter(-9.537e-4,0)
#plt.plot(xn[:,2],xn[:,3])

rotation_coords = np.column_stack((xn[:,2],xn[:,3]))
derotated_coords = derotate(rotation_coords,times)
plt.plot(derotated_coords[:,0],derotated_coords[:,1],label='Non-Rotating Orbit')
plt.scatter(derotated_coords[-1,0],derotated_coords[-1,1])
plt.show()