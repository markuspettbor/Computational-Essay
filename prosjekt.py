import numpy as np
import matplotlib.pyplot as plt
from numba import jit #This speeds up the simulation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from numpy.linalg import norm

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
'''
x = np.arange(-10.0, 10.0, .1) #create a grid of points from z = -10 to 10
y = np.arange(-10.0, 10.0, .1) #create a grid of points from y = -10 to 10
z = np.arange(-10.0, 10.0, .1) #create a grid of points from z = -10 to 10
X, Y, Z = np.meshgrid(y,z,x) #turn this into a mesh
ilen, jlen, klen = np.shape(Y) #define the length of the dimensions, for use in iteration
Bf = np.zeros((ilen,jlen,klen,3)) #set the points to 0


for i in range(0, ilen): #iterate through the grid, setting each point equal to the magnetic field value there
    print(i, 'of', ilen)
    for j in range(0, jlen):
        for k in range(0, klen):
            Bf[i,j,k] = B_bot(X[i,j,k], Y[i,j,k], Z[i,j,k])

plt.streamplot(Y,Z, Bf[:,:,1], Bf[:,:,2]) #plot the magnetic field
plt.xlim(-10.0,10.0)
plt.ylim(-10.0,10.0)
plt.xlabel("$y$")
plt.ylabel("$z$")
plt.title("Magnetic Field of a 'Magnetic Bottle'")
plt.show()
'''
'''
m = 4.0*m_p #mass of the alpha particle
q = 2.0*qe #charge of the alpha particle
QoverM = q/m

scale = 1
dt = 1e-4/scale #small timestep

t = np.arange(0, 100/scale, dt) #create an array that will hold the times
rp = np.zeros((len(t), 3)) #create an array that will hold the position values
vp = np.zeros((len(t), 3)) #create an array that will hold the velocity values

v0 = 10
rp[0,:] = np.array([0, -5, 1]) #initialize the position to y=-5, 5m above the lower dipole
vp[0,:] = np.array([-v0,  0, 0*v0/10]) #initialize the velocity to be in the z-direction

ei = -1
for it in np.arange(0, len(t)-1,1):
    Bp = B_bot(rp[it,0], rp[it, 1], rp[it,2])*scale #input the current particle position into the B_bot function to get the magnetic field
    Ap = QoverM * np.cross(vp[it,:], Bp) #Calculate the magnetic force on the particle
    vp[it+1] = vp[it] + dt*Ap #Update the velocity of the particle based on this force
    rp[it+1] = rp[it] + dt*vp[it] #Update the positon of the particle based on this velocity
    if (np.sqrt(np.sum(rp[it+1]**2)) > 20.0): #If the particle escapes (goes more than 20m away from the origin) end the loop
        print('particle escapes at t = %f' %(t[it]))
        ei = it
        break

fig  = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot(rp[:ei,0], rp[:ei,1], rp[:ei,2])
#ax.streamplot(X, Y, Z, Bf[:,:,0], Bf[:,:,1], Bf[:,:,2], color="black")
ax.scatter(rp[ei,0], rp[ei,1], rp[ei,2], c = 'r')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

'''
'''
# now to make different views of the charged particle's trajectory
#plt.streamplot(Y,Z, Bf[:,:,1], Bf[:,:,2], color="black")
plt.plot(rp[:,1], rp[:,2])
plt.xlim(-10.0,10.0)
plt.ylim(-10.0,10.0)
plt.xlabel("$y$")
plt.ylabel("$z$")
plt.title("Trajectory of Alpha Particle in a 'Magnetic Bottle'")
plt.show()
'''
# Here begins our stuff
'''
def dB(a, b, z, phi, dphi, theta = 0):
    r = np.sqrt(a**2 +b**2 + z**2 - 2*a*b*np.cos(phi - theta))
    dB =  mu0*I/(4*np.pi*r**3)*(a**2 - a*b*np.cos(phi - theta))*dphi
    return 2*dB

m_p = 1.67E-27        # mass of proton: kg
qe = 1.602E-19        # charge of proton: C
mu0 = np.pi * 4.0E-7  # mu_naught
a = 3
z = 1
I = 1000 # foob
theta = 0
steps = 1000
steps2 = 1000
dphi = 2*np.pi/steps
b_lin = np.linspace(0, 10, steps2)
B = np.zeros(steps2)
phi = 0

for k in range(steps2):
    sum = 0
    for i in range(steps):
        phi = phi + dphi
        sum += dB(a, b_lin[k], z, phi, dphi, theta)
    B[k] = sum
plt.plot(b_lin, B)
print('B field at z = %f: %e' %(z, sum))
print('B field from one coil: ', sum/2)  # Matches equivalent thing from hyperphysics calculator :)
# http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html
plt.grid(True)
plt.show()
# mv^2/b = qvB(I)
# v = v0 + qvB*dt

'''
# Here there be memes

@jit(nopython = True)
def dBeta(a, b, z, phi, dphi, theta = 0):
    r = np.sqrt(a**2 +b**2 + z**2 - 2*a*b*np.cos(phi - theta))
    dBeta =  mu0/(4*np.pi*r**3)*(a**2 - a*b*np.cos(phi - theta))*dphi
    return 2*dBeta

def Beta_func():
    m_p = 1.67E-27        # mass of proton: kg
    qe = 1.602E-19        # charge of proton: C
    c = 299792458
    mu0 = np.pi * 4.0E-7  # mu_naught
    a = 100 #10 cm
    z = 5
    I = 1000000 # foob
    theta = 0
    steps = 1000
    steps2 = 1000
    dphi = 2*np.pi/steps
    b_lin = np.linspace(0, 200, steps2)
    phi = 0
    time = np.linspace(0,5,2000)
    dt = time[1] - time[0]
    Beta = np.zeros(time.shape)

    pos = np.zeros([len(time), 3])
    vel = np.zeros(pos.shape)
    steppi = 20
    moddus = int(len(time)/steppi)
    pos[0] = np.array([50, 50, z/2])
    vel[0] = np.array([0, 1000, 0])
    r = 50
    print(r)
    for k in range(len(time)-1):
        r = norm(pos[k,:-1])
        sum = 0
        phi = 0
        for i in range(steps):
            phi = phi + dphi
            sum += dBeta(a, r, z, phi, dphi, theta)
        Beta[k] = sum
        I_need = m_p * 1/np.sqrt(1/norm(vel[k])**2 - 1/c**2) / (qe * r * Beta[k])
        #I_need = 10
        #print(norm(vel[k]))
        acc = np.cross(qe*vel[k]*I_need/m_p , np.array([0,0,Beta[k]]))
        vel[k+1] = vel[k] - acc*dt
        pos[k+1] = pos[k] + vel[k+1]*dt
        if k%moddus == 0:
            print('%i of %i' %(k/moddus, steppi))
    plt.plot(pos[:,0], pos[:,1])
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
'''
    #Beta_Interpol = interp1d(b_lin, Beta, kind = 'quadratic')

    #v = c*0.999999999999

    #I_need = m_p * 1/np.sqrt(1/v**2 - 1/c**2) / (qe * r * Beta_Interpol(r))

    print('I needed', I_need)
    plt.plot(b_lin, Beta_Interpol(b_lin))
    print('B field at z = %f: %e' %(z, sum))
    print('B field from one coil: ', sum/2)  # Matches equivalent thing from hyperphysics calculator :)
    # http://hyperphysics.phy-astr.gsu.edu/hbase/magnetic/curloo.html
    plt.grid(True)
    plt.show()
'''
Beta_func()
