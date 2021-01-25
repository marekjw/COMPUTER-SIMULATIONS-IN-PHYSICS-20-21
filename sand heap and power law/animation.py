from sand_heap import SandHeap
import matplotlib.animation as animation
from sand_heap import run_simulation
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(10, 10))

heap = SandHeap(52, 3)

ax = fig.add_subplot(111)
ax.set_title('Height of the Sandpile')
cax = ax.imshow(heap.get_array(), cmap='plasma')

cax.set_clim(vmin=0, vmax=4)
cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3, 4], orientation='vertical')

idx_middle = (26, 26)


def init_func():
    heap.reset()
    cax.set_data(heap.get_array())
    return cax,


def animate(_):
    heap.drop_grain(idx_middle)
    cax.set_data(heap.get_array())

    return cax,


ani = animation.FuncAnimation(
    fig, animate, frames=500, blit=True, interval=200)

plt.show()
