from wator import TheWorld
import matplotlib.animation as animation
import matplotlib.pyplot as plt

x = 40
y = 40
multiply_fish = 3
multiply_shark = 20
shark_energy = 3
n_fish = 300
n_sharks = 10


fig = plt.figure(figsize=(10, 10))


def create_world():
    return TheWorld(
        x, y,
        n_fish=n_fish,
        n_sharks=n_sharks,
        multiply_fish=multiply_fish,
        multiply_sharks=multiply_shark,
        sharks_energy=shark_energy
    )


world = create_world()

ax = fig.add_subplot(111)
ax.set_title('The World')

cax = ax.imshow(world.to_numpy_array(), cmap='bwr')
cax.set_clim(vmin=-1, vmax=1)


def init_func():
    world = create_world()
    cax.set_data(world.to_numpy_array())
    return cax,


def animate(_):
    world.step()
    cax.set_data(world.to_numpy_array())

    return cax,


ani = animation.FuncAnimation(
    fig, animate, frames=500, blit=True, interval=10
)

plt.show()
