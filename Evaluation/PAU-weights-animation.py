import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Read coefficients from CSV file
a_coefficients = []
b_coefficients = []
file = "weights2x1024.csv"

with open(file, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for i, row in enumerate(datareader):
        if i == 0 or 0 != i%2000:  # skip header row
            continue
        if row[1]=="0":
            a_coefficients.append(eval(row[0])[1][0][0])
            b_coefficients.append(eval(row[0])[1][1][0])

print(len(a_coefficients))
# Check if coefficients are available
if not a_coefficients or not b_coefficients:
    print("No coefficients found in the CSV file.")
    exit()

# Define the function R(x)
def R(x, a_coeffs, b_coeffs):
    numerator = sum(a * x**j for j, a in enumerate(a_coeffs))
    denominator = 1 + sum(b * x**j for j, b in enumerate(b_coeffs, start=1))
    return numerator / denominator

x_values = np.linspace(-5, 5, 100)  # Adjust range as needed

# Create a figure and axis object
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
frame_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top', fontsize=12)

ax.set_xlabel('x')
ax.set_ylabel('R(x)')
ax.set_title('Plot of R(x)')
ax.grid(True, linestyle='--', linewidth=1.5)  # Thick dashed grid lines
ax.set_xlim(-5,5)
ax.set_ylim(-1,2)
# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    frame_text.set_text('')
    return line,

# Animation function: this is called sequentially
def animate(i):
    y_values = [R(x, a_coefficients[i], b_coefficients[i]) for x in x_values]
    line.set_data(x_values, y_values)
    frame_text.set_text('Environment: {}'.format(i*2000))

    return line,

# Create the animation
ani = FuncAnimation(fig, animate, frames=len(a_coefficients), init_func=init, blit=True)
ani.save('animation.gif', writer='pillow', fps=20)
