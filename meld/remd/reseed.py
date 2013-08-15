import random


class NullReseeder(object):
    '''
    Dummy reseeder that does nothing.
    '''
    def reseed(self, time, current_states, store):
        pass


class Reseeder(object):
    '''
    Reseed replicas from previous states.

    This class implements a reseeder that will periodically reseed all replicas with
    structures from the past history of the lowest replica.
    '''
    def __init__(self, interval, candidate_frames):
        '''
        Initialize the reseeder.

        :param interval: number of timesteps between reseeding
        :param candidate_frames: consider this many frames previous to the current frame for reseeding
        '''
        self.interval = interval
        self._next_reseed = interval
        self.candidate_frames = candidate_frames

        assert self.interval > self.n_replicas
        assert self.candidate_frams < (self.interval - self.n_replicas)

    def reseed(self, step, current_states, store):
        '''
        Perform the reseeding.

        :param step: the current timestep
        :param current_states: a list of the current replica states to be modified
        :param store: a DataStore object to get historical structures from
        '''
        if step == self._next_reseed:
            self._next_reseed += self.interval
            for state in current_states:
                from_frame = random.randrange(step - self.candidate_frames, step)
                all_coords = store.load_positions_random_access(from_frame)
                state.positions = all_coords[0, :, :]
