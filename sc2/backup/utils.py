from collections import namedtuple, deque


Transition = namedtuple('Transition', ('state', 'action', 'log_action_prob', 'reward'))
class Buffer:
    def __init__(self, size=8):
        self.maxlen = size
        self.memory = deque(maxlen=size)

    def append(self, experience):
        self.memory.append(Transition(*experience))

    def is_full(self):
        return len(self.memory) == self.maxlen

    def reset(self):
        self.memory.clear()

    def get_reversed_experience(self):
        return reversed(self.memory)

    def get_n_steps_data(self):
        return Transition(*zip(*self.memory))


def freeze_layers(layer):
    for param in layer.parameters():
        param.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)