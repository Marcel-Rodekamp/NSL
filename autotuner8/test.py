from _functions import *
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Data for plotting
xs = np.linspace(0, 100, 101)
# Initial parameters: [a0, a1, a2]
init_params = [0, 1, 0.5]
ys = model_to_fit(init_params, xs)

# Create figure and the line that we will update
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.set_ylim(0, 1)
l, = plt.plot(xs, ys, lw=2)
ax.set_title("Interactive Fit")

# Define slider axes and create sliders
ax_a0 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_a1 = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_a2 = plt.axes([0.25, 0.05, 0.65, 0.03])

slider_a0 = Slider(ax_a0, 'Param a0', -100.0, 100.0, valinit=init_params[0])
slider_a1 = Slider(ax_a1, 'Param a1', -10.0, 10.0, valinit=init_params[1])
slider_a2 = Slider(ax_a2, 'Param a2', -5.0, 5.0, valinit=init_params[2])

# Update function: read slider values and update the plot
def update(val):
    new_params = [slider_a0.val, slider_a1.val, slider_a2.val]
    new_ys = model_to_fit(new_params, xs)
    l.set_ydata(new_ys)
    fig.canvas.draw_idle()

slider_a0.on_changed(update)
slider_a1.on_changed(update)
slider_a2.on_changed(update)

plt.show()