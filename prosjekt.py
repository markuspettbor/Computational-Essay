import numpy as np
import matplotlib.pyplot as plt
from numba import jit #This speeds up the simulation
from mpl_toolkits.mplot3d import Axes3D

# Parameters for plot attributes
plt.rc("xtick", labelsize="large")
plt.rc("ytick", labelsize="large")
plt.rc("axes", labelsize="xx-large")
plt.rc("axes", titlesize="xx-large")
plt.rc("figure", figsize=(8,8))

# define key constants
m_p = 1.67E-27       # mass of proton: kg
qe = 1.602E-19        # charge of proton: C
mu0 = np.pi * 4.0E-7 #mu_naught
mu = 10000.0 * np.array([0.0, 0.0, 1.0]) # magnetic moment that points in the z direction

# The jit command ensures fast execution using numba
@jit
def B_bot(x,y,z):
    zdisp = 10.0 #displacement of the two magnetic dipoles away from zero (one is at z = +zdisp, the other at -zdisp)

    # point dipole A
    posA = np.array([0.0, 0.0, zdisp]) #set the position of the first dipole
    rA = np.array([x,y,z]) - posA #find the difference between this position and the observation position
    rmagA = np.sqrt(sum(rA**2))
    B1A = 3.0*rA*np.dot(mu,rA) / (rmagA**5) #calculate the contribution of the first term to the magnetic field
    B2A = -1.0 * mu / (rmagA**3) #calculate the contribution of the second term

    # point dipole B
    posB = np.array([0.0, 0.0, -zdisp])
    rB = np.array([x,y,z]) - posB
    rmagB = np.sqrt(sum(rB**2))
    B1B = 3.0*rB*np.dot(mu,rB) / (rmagB**5)
    B2B = -1.0 * mu / (rmagB**3)

    return ((mu0/(4.0*np.pi)) * (B1A + B2A + B1B + B2B)) #return the magnetic field due to the magnetic bottle.

y = np.arange(-10.0, 10.0, .1) #create a grid of points from y = -10 to 10
z = np.arange(-10.0, 10.0, .1) #create a grid of points from z = -10 to 10
Y, Z = np.meshgrid(y,z) #turn this into a mesh
ilen, jlen = np.shape(Y) #define the length of the dimensions, for use in iteration
Bf = np.zeros((ilen,jlen,3)) #set the points to 0


for i in range(0, ilen): #iterate through the grid, setting each point equal to the magnetic field value there
    for j in range(0, jlen):
        Bf[i,j] = B_bot(0.0, Y[i,j], Z[i,j])
'''
plt.streamplot(Y,Z, Bf[:,:,1], Bf[:,:,2]) #plot the magnetic field
plt.xlim(-10.0,10.0)
plt.ylim(-10.0,10.0)
plt.xlabel("$y$")
plt.ylabel("$z$")
plt.title("Magnetic Field of a 'Magnetic Bottle'")
plt.show()
'''
m = 4.0*m_p #mass of the alpha particle
q = 2.0*qe #charge of the alpha particle
QoverM = q/m

dt = 1e-4 #small timestep

t = np.arange(0, 1, dt) #create an array that will hold the times
rp = np.zeros((len(t), 3)) #create an array that will hold the position values
vp = np.zeros((len(t), 3)) #create an array that will hold the velocity values

v0 = 400
rp[0,:] = np.array([0, -5, 1]) #initialize the position to y=-5, 5m above the lower dipole
vp[0,:] = np.array([-v0,  0, 0*v0/10]) #initialize the velocity to be in the z-direction

for it in np.arange(0, len(t)-1,1):
    Bp = B_bot(rp[it,0], rp[it, 1], rp[it,2]) #input the current particle position into the B_bot function to get the magnetic field
    Ap = QoverM * np.cross(vp[it,:], Bp) #Calculate the magnetic force on the particle
    vp[it+1] = vp[it] + dt*Ap #Update the velocity of the particle based on this force
    rp[it+1] = rp[it] + dt*vp[it] #Update the positon of the particle based on this velocity
    if (np.sqrt(np.sum(rp[it+1]**2)) > 20.0): #If the particle escapes (goes more than 20m away from the origin) end the loop
        break

fig  = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot(rp[:,0], rp[:,1], rp[:,2])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
'''
# now to make different views of the charged particle's trajectory
#plt.streamplot(Y,Z, Bf[:,:,1], Bf[:,:,2], color="black")
plt.plot(rp[:,1], rp[:,2])
plt.xlim(-10.0,10.0)
plt.ylim(-10.0,10.0)
plt.xlabel("$y$")
plt.ylabel("$z$")
plt.title("Trajectory of Alpha Particle in a 'Magnetic Bottle'")
'''
plt.show()
