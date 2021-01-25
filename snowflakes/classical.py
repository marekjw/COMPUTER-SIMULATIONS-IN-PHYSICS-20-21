from particles import AggregateClassical
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fastprogress import progress_bar


n_particles = 1000


def run_simulation(n_particles=10000, draw_every=50):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-100, 100), ylim=(-100, 100))
    agg = AggregateClassical()

    ims = []

    for particle in progress_bar(range(n_particles)):
        agg.step()
        if particle % draw_every == 0:
            ims.append(ax.plot(*zip(*agg.get_aggregate()),
                               'o', markersize=2, color='b'))

    im_ani = animation.ArtistAnimation(fig, ims, interval=120, repeat_delay=1500,
                                       blit=True)
    # im_ani.save('im.mp4', metadata={'artist': 'Guido'})

    plt.show()


run_simulation(n_particles)
