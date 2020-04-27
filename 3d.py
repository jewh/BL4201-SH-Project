
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Define your continous function to plot
# for very rough - f= x/np.sin(np.sqrt(x**2 + y**2))
# for bumpy surface cos(ax) + cos(ay) seems to work nicely (sin too)
# For smooth search space f = np.e**-(x**2 + y**2)

def smooth(x,y):
    f = (np.e**-(5.0*(x**2 + y**2)))
    # limits of +- 2 work best here
    return f

def rough(x,y):
    f = x/np.sin(np.sqrt(x**2 + y**2))
    f = abs(f)
    # limits of +- 10 work best
    return f

def plateau(x,y):
    f = abs(np.cos(6*x) + np.cos(6*y)+6)*(x**2 + y**2 + 1)/(np.e**(x**2 + y**2)+1)
    # limits of +- 3 work best here
    return f


# Now plot
# Define a resolution
# NOTE - this is very slow for number > 100
number = 100

# define the figure

fig = plt.figure()
# define colour scheme
colour = 'winter'
# ______________________________________________________________________________________________________________________
# Now define the first figure
ax1 = fig.add_subplot(3,1,1, projection="3d")

# Now define the x and y limits
x1_min = -1.5
x1_max = 1.5
y1_min = -1.5
y1_max = 1.5
# Now the axes
x = np.linspace(x1_min, x1_max, number)
y = np.linspace(y1_min, y1_max, number)
X,Y = np.meshgrid(x,y)
Z = smooth(X,Y)

ax1.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap = colour, edgecolor='none')
ax1._axis3don = False
ax1.set_title("A", loc="left")
# ______________________________________________________________________________________________________________________
# Now the second figure
ax2 = fig.add_subplot(3,1,2,projection="3d")
# Now define the x and y limits
x2_min = -5
x2_max = 5
y2_min = -5
y2_max = 5
# Now the axes
x = np.linspace(x2_min, x2_max, number)
y = np.linspace(y2_min, y2_max, number)
X,Y = np.meshgrid(x,y)
Z = rough(X,Y)

ax2.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap = colour, edgecolor='none')
ax2._axis3don = False
ax2.set_title("B", loc="left")
# ______________________________________________________________________________________________________________________
# Now the third plot
ax3 = fig.add_subplot(3,1,3, projection="3d")
x3_min = -3.5
x3_max = 3.5
y3_min = -3.5
y3_max = 3.5
# Now the axes
x = np.linspace(x3_min, x3_max, number)
y = np.linspace(y3_min, y3_max, number)
X,Y = np.meshgrid(x,y)
Z = plateau(X,Y)

ax3.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap = colour, edgecolor='none')
ax3._axis3don = False
ax3.set_title("C", loc="left")

# Now set overall title
fig.suptitle("Search Space Topology", fontsize=14, weight="bold")
plt.savefig("C:/Users/User/Documents/GitHub/BL4201-SH-Project/searchspace3d.png", bbox_inches='tight')
plt.show()

