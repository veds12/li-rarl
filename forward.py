from pydreamer.models.dreamer import Dreamer

_forward_dict = {"dreamer": Dreamer}


def get_forward_module(name):
    return _forward_dict[name]
