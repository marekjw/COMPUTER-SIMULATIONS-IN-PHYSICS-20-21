from abc import abstractmethod
import numpy as np

move_compass = [
    (0, 1),
    (1, 0),
    (-1, 0),
    (0, -1)
]


class Particle:

    def __init__(self, x, y) -> None:
        self.x, self.y = int(x), int(y)

    def random_move(self):
        dx, dy = move_compass[np.random.randint(0, 4)]
        self.x += dx
        self.y += dy

    def get_position(self):
        return self.x, self.y

    def move_to(self, x, y):
        self.x, self.y = x, y


class AgregateAbstract:
    def __init__(self, add_spawn_radius=3, add_kill_radius=10) -> None:
        self.aggregate = set([(0, 0)])
        self.aggregate_radius = 1
        self.add_spawn_radius = add_spawn_radius
        self.add_kill_radius = add_kill_radius
        self.adjust_radius()
        self.particle = None

    def adjust_radius(self):
        self.spawn_radius = self.aggregate_radius + self.add_spawn_radius
        self.kill_radius = self.spawn_radius + self.add_kill_radius

    def new_particle(self):
        theta = np.random.random() * 2 * np.pi

        self.particle = Particle(
            self.spawn_radius * np.cos(theta), self.spawn_radius*np.sin(theta))

    @abstractmethod
    def handle_touch(self):
        raise NotImplementedError()

    def touches(self, position):
        x, y = position
        for dx, dy in move_compass:
            if (x + dx, y + dy) in self.aggregate:
                return True
        return False

    def move_until_touches(self) -> None:
        self.new_particle()
        while not self.touches(self.particle.get_position()):

            if np.linalg.norm(self.particle.get_position()) > self.kill_radius:
                self.new_particle()

            self.particle.random_move()

    def get_aggregate(self):
        return self.aggregate

    def set_radius(self, position):
        radius = int(np.linalg.norm(position))
        if radius > self.aggregate_radius:
            self.aggregate_radius = radius
            self.adjust_radius()

    def step(self):
        self.move_until_touches()
        self.handle_touch()


class AggregateClassical(AgregateAbstract):
    def __init__(self, add_spawn_radius=3, add_kill_radius=10) -> None:
        super().__init__(add_spawn_radius, add_kill_radius)

    def handle_touch(self) -> None:
        self.aggregate.add(self.particle.get_position())
        self.set_radius(self.particle.get_position())
        self.particle = None


class AgregateProbability(AgregateAbstract):

    class CannotMoveParticle(Exception):
        pass

    def __init__(self, probability, add_spawn_radius=3, add_kill_radius=10) -> None:
        super().__init__(add_spawn_radius=add_spawn_radius, add_kill_radius=add_kill_radius)
        self.p = probability

    def step(self):
        while 1:
            self.move_until_touches()
            if self.handle_touch():
                return

    def handle_touch(self):
        '''
        return true if particle sticked to the aggregate,
        moves particle to a random free adjacent field and returns False otherwise
        '''
        if np.random.random() < self.p:
            self.aggregate.add(self.particle.get_position())
            self.set_radius(self.particle.get_position())
            return True

        x, y = self.particle.get_position()
        for dx, dy in np.random.permutation(move_compass):
            if not (x + dx, y + dy) in self.aggregate:
                self.particle.move_to(x+dx, y + dy)
                return False

        raise AgregateProbability.CannotMoveParticle()
