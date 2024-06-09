import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from argparse import ArgumentParser



# Define the function R(x)
def R(x, a_coeffs, b_coeffs):
    numerator = sum(a * x**j for j, a in enumerate(a_coeffs))
    denominator = 1 + sum(b * x**j for j, b in enumerate(b_coeffs, start=1))
    return numerator / denominator
def init():
    line.set_data([], [])
    frame_text.set_text('')
    return line,
def animate(i):
    y_values = [R(x, a_coefficients[i], b_coefficients[i]) for x in x_values]
    line.set_data(x_values, y_values)
    frame_text.set_text('Environment: {}'.format(i*2000))
    return line,


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", help="insert file path", type=str)
    parser.add_argument("--network", help="of which network (if CDE is used)should the layers be analyzed: starting at 0", default="0", type=str)
    parser.add_argument("--layer", help="layer which should by analyzed: starting at 0", default=0, type=int)
    parser_args = parser.parse_args()

    a_coefficients = []
    b_coefficients = []
    with open(parser_args.file, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for i, row in enumerate(datareader):
            if i == 0 or 0 != i%2000:  # skip header row
                continue
            if row[1]==parser_args.network:
                a_coefficients.append(eval(row[0])[parser_args.layer][0][0])
                b_coefficients.append(eval(row[0])[parser_args.layer][1][0])

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


    # Create the animation
    ani = FuncAnimation(fig, animate, frames=len(a_coefficients), init_func=init, blit=True)
    ani.save(parser_args.file[:-4] + '_animation.gif', writer='pillow', fps=20)
