from sand_heap import SandHeap
import matplotlib.animation as animation
from sand_heap import run_simulation
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(10, 10))

heap = SandHeap(50, 3, supercritical=True)

ax = fig.add_subplot(111)
ax.set_title('Height of the Sandpile')
cax = ax.imshow(heap.get_array(), cmap='plasma')

cax.set_clim(vmin=0, vmax=8)
cbar = fig.colorbar(cax, ticks=[0, 3, 5, 8], orientation='vertical')


def animate(_):
    heap.just_cascade()
    cax.set_data(heap.get_array())

    return cax,


ani = animation.FuncAnimation(
    fig, animate, frames=500, blit=True, interval=10)

plt.show()
