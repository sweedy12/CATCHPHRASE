import numpy as np

def get_positive(x):

    if x < 0:
        print(0)
        return 0
    print(x)
    return float(x)


g = np.vectorize(get_positive)

vec = np.array([np.array([-0.3]),np.array([0.2]),np.array([-1]),np.array([5])], dtype=np.float32)
# print(g(vec))

vec[np.argwhere(vec <0)] =0
print(vec)
