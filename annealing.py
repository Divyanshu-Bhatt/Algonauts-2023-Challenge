class LinearAnnealing(object):
    """Linear annealing from 0 to 1 over a given number of epochs.
    For the Beta VAE, annealing_epochs"""
    def __init__(self, annealing_epochs):
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0

    def __call__(self):
        self.current_epoch += 1
        return min(1, self.current_epoch / self.annealing_epochs)


class CyclicAnnealing(object):
    """Linear annealing from 0 to 1 over a given number of epochs, then back to 0.
    Repeat this process every 2 * annealing_epochs epochs.
    For the Beta VAE, annealing_epochs"""
    def __init__(self, annealing_epochs):
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0
        self.restart_epoch = 0

    def __call__(self):
        self.current_epoch += 1
        if self.current_epoch > 2 * self.annealing_epochs:
            self.current_epoch = 0
            self.restart_epoch += 1
        return min(1, self.current_epoch / self.annealing_epochs)
