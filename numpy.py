class ndarray:
    pass

def array(x):
    return x

class random:
    @staticmethod
    def random(shape):
        if isinstance(shape, tuple):
            if len(shape) == 2:
                return [[0 for _ in range(shape[1])] for _ in range(shape[0])]
            return [0 for _ in range(shape[0])]
        return [0]

def where(condition):
    return []

def argsort(arr):
    return []

class linalg:
    @staticmethod
    def norm(x, axis=None):
        return 0
