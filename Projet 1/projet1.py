import numpy as np
import matplotlib.pyplot as plt


### Parameters ###

L = 0.8
a = 0.2
e = 0.15
t = 0.01
Mass = 188 * 0.4
Mg = (188*0.4) * 9.81
P = 1000
E = 210000 * (1000**2)  # 210000 MPa * 1000^2 pour passer en N/m2
I = (e**4-(e-2*t)**4)/12
I_hollowDisk = np.pi/32 * (e**4 - (e-2*t)**4)

I_ratio = np.sqrt(I_hollowDisk/I)
EI = E*I




### Moment computation ###

def M(x):
    if x < 0 or x > 1.2:
        return 0
    if x <= L:
        Moment = -2*a*P #M_AC
    elif x <= L+2*a:
        Moment = ((x-L)-2*a)*P #M_CE

    return Moment


### Displacement computation ###

# as in the formula for deflection [d2z/dx2 = - M/EI],
# z is oriented downwards,
# we will use [d2z/dx2 = + M/EI] instead

z0 = (a*P)/(EI) * L**2

omegaC = - (2*a*P)/(EI) * L


def flecheAC(x):
    return (-a*P)/EI * x**2 + z0


def flecheCE(x):
    x = x-L
    return (-a*P*x**2 + P*(x**3)/6)/EI + omegaC * x


def fleche(x):
    if x <= L:
        return flecheAC(x)
    elif x <= L + 2*a:
        return flecheCE(x)



### Plot of the displacements ###

# Create the plot
fig, ax = plt.subplots(figsize=(7, 4 * 7/8))

# Define the origin for both vectors (the arrows will start at (0, 0))
origin_x = [0, 0]
origin_y = [0, 0]

# Define the components of the vectors for the X and Z directions
vector_x = [0.1, 0]  # X-axis vector: (0.1, 0)
vector_z = [0, 1.5e-5]  # Z-axis vector: (0, 2e-5)

# Use quiver to plot the arrows (datum system)
ax.quiver(origin_x, origin_y, vector_x, vector_z, angles='xy', scale_units='xy', scale=1, color=['blue', 'blue'],zorder = 3)

plt.text(0.105, 0, 'X', fontsize=10, color='blue')  # X-axis label
plt.text(0, 1.55e-5, 'Z', fontsize=10, color='blue')  # Z-axis label



n = 1000
start = 0
end = 1.2
xplot = np.linspace(start, end, n, endpoint=True)
yplot = np.empty(n)
for i in range(n):
    k = (end-start)/n
    yplot[i] = fleche(start + i*k)


# Improve axes appearance
ax.spines['top'].set_visible(False)  # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine
ax.spines['left'].set_position(('outward', 10))  # Offset left spine
ax.spines['bottom'].set_position(('outward', 10))  # Offset bottom spine

plt.grid(True,alpha=0.7)  # Add grid for better visualization

# Set axis limits based on data range
plt.xlim(0, 1.2)
plt.ylim(-4e-5, 4e-5)


plt.plot(xplot, yplot, label = "Configuration déformée", color = "red")
yplot = np.zeros_like(xplot)
plt.plot(xplot, yplot, label = "Configuration initiale", color = "black")
plt.scatter(0.8, 0, color='k', zorder=3, s=10)

#plt.title("Déflexion du modèle sous une charge de 1000N ", fontsize=14, fontweight='bold', pad=20)
plt.legend(loc='lower left')
plt.show()
plt.savefig('deflectionProjet1.jpg', dpi=300) 


### Computation of the elastic potential energy ###

U_AC = 1/2 * (2*a*P)**2/EI * L

#U_CE = 1/2 * (P**2/(EI)) * (2*a*L**2 - 4*a**2*L + 8/3*a**3)
U_CE = 1/2 * (P**2/(EI)) * (8/3*a**3)

#U_tot = (P**2/(EI)) * (2*a**2 *L + 4/3 * a**3)
U_tot = U_AC + U_CE

k_eq = 2 * U_tot / (z0 ** 2)

k_real = P/z0

k_rapp = k_eq/k_real


### Computation of the equivalent Mass ###

m_eq = (fleche(L/2)/fleche(0))**2 * Mass


### Computation of the natural Frequency of the model ###

freq_n = (k_eq/m_eq)**(1/2) / (2* np.pi)
