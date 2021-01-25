from abc import abstractmethod
import numpy as np


class Animal:
    def __init__(self, idx,  world, multiply_every):
        self.idx = idx
        self.prievious_idx = idx
        self.world = world
        self.multiply_every = multiply_every
        self.counter = 0

    @abstractmethod
    def move(self):
        raise NotImplementedError()

    def step_callback(self):
        pass

    def step(self):
        self.prievious_idx = self.idx
        idx = self.move()

        self.counter += 1

        if self.counter % self.multiply_every == 0:
            if idx is not None:
                self.multiply()
                self.counter = 0
            else:
                self.counter -= 1

        self.step_callback()


class Fish(Animal):
    def __init__(self, idx, world, multiply):
        super().__init__(idx, world, multiply)

    def move(self):
        res = self.world.get_free_adjacent(self.idx)
        if res is not None:
            self.world.move_fish(self.idx, res)
            self.idx = res
        return res

    def multiply(self):
        self.world.create_fish(self.prievious_idx)


class Shark(Animal):

    class StarvationException(Exception):
        def __init__(self, idx):
            self.idx = idx

    def __init__(self, idx, world, multiply_every, energy):
        super().__init__(idx, world, multiply_every)
        self.energy_max = energy
        self.energy = energy

    def move(self):
        res = self.world.get_adjacent_fish(self.idx)

        self.energy -= 1

        if res is not None:
            self.energy = self.energy_max
            self.world.kill_fish(res)
            self.world.move_shark(self.idx, res)

            self.idx = res
            return res

        res = self.world.get_free_adjacent(self.idx)

        if res is not None:
            self.world.move_shark(self.idx, res)
            self.idx = res

        return res

    def step_callback(self):
        if self.energy == 0:
            raise Shark.StarvationException(self.idx)

    def multiply(self):
        self.world.create_shark(self.prievious_idx)


class TheWorld:
    # is this a jojo reference?

    shark_code = 1
    fish_code = -1
    water_code = 0

    move_compass = [
        (0, 1),
        (0, -1),
        (-1, 0),
        (1, 0)
    ]

    def __init__(self, x_dim, y_dim, n_fish, n_sharks, multiply_fish, multiply_sharks, sharks_energy) -> None:
        self.fishes = {}
        self.sharks = {}
        self.sharks_energy = sharks_energy
        self.multiply_sharks = multiply_sharks
        self.multiply_fish = multiply_fish
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.map = np.zeros((x_dim, y_dim))

        for _ in range(n_fish):
            self.create_fish(self.get_random_free())

        for _ in range(n_sharks):
            self.create_shark(self.get_random_free())

    def field_is_free(self, idx):
        return self.map[idx] == TheWorld.water_code

    def get_random_free(self):
        while 1:
            idx = np.random.randint(
                0, self.x_dim), np.random.randint(0, self.y_dim)
            if self.field_is_free(idx):
                return idx

    def step(self):
        for idx in list(self.fishes.keys()):
            self.fishes[idx].step()

        for idx in list(self.sharks.keys()):
            try:
                self.sharks[idx].step()
            except Shark.StarvationException as Starved:
                self.kill_shark(Starved.idx)

    def move_fish(self, prvs_field, to):
        assert prvs_field in self.fishes, 'Cannot move fish - no fish on given field'
        assert self.field_is_free(
            to), 'Cannot move fish - target field is not free'

        moved_fish = self.fishes.pop(prvs_field)
        self.fishes[to] = moved_fish
        self.map[prvs_field] = TheWorld.water_code
        self.map[to] = TheWorld.fish_code

    def move_shark(self, prvs_field, to):
        assert prvs_field in self.sharks, 'Cannot move shark - no shark on given field'
        assert self.field_is_free(
            to), 'Cannot move shark - target field is not free'

        moved_shark = self.sharks.pop(prvs_field)
        self.sharks[to] = moved_shark
        self.map[prvs_field] = TheWorld.water_code
        self.map[to] = TheWorld.shark_code

    def get_free_adjacent(self, idx):
        x, y = idx
        for dx, dy in np.random.permutation(TheWorld.move_compass):
            nx, ny = (x + dx) % self.x_dim, (y + dy) % self.y_dim
            if self.field_is_free((nx, ny)):
                return nx, ny
        return None

    def get_adjacent_fish(self, idx):
        x, y = idx
        for dx, dy in np.random.permutation(TheWorld.move_compass):
            nx, ny = (x+dx) % self.x_dim, (y+dy) % self.y_dim
            if self.map[nx, ny] == TheWorld.fish_code:
                return nx, ny
        return None

    def kill_fish(self, idx):
        assert idx in self.fishes, 'Cannot kill fish - no fish on given field'
        del self.fishes[idx]
        self.map[idx] = TheWorld.water_code

    def kill_shark(self, idx):
        assert idx in self.sharks, 'Cannot kill shark - no shark on given field'
        del self.sharks[idx]
        self.map[idx] = TheWorld.water_code

    def create_shark(self, idx):
        assert self.field_is_free(idx), 'Cannot create shark on non-free field'
        self.sharks[idx] = Shark(
            idx, self, self.multiply_sharks, self.sharks_energy)
        self.map[idx] = TheWorld.shark_code

    def create_fish(self, idx):
        assert self.field_is_free(idx), 'Cannot create fish on non-free field'
        self.fishes[idx] = Fish(idx, self, self.multiply_fish)
        self.map[idx] = TheWorld.fish_code

    def to_numpy_array(self):
        return self.map
