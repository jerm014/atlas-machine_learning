
import numpy as np


P = np.array([[ .6,.39, .01,   0,   0],
              [ .2, .5,  .3,   0,   0],
              [.01,.24,  .5, .24, .01],
              [  0,  0, .15,  .7, .15],
              [  0,  0, .01, .39,  .6]])

IS = np.array([.01,.04,.03,.02,.9])
print(f"initial stste: {IS}")
last = IS

for i in range(1000):
    IS = IS @ P
    print(f"{i}            : {IS}")
    # compare if two numpy ndarrays
    if np.array_equal(last, IS):
        print(f"converged at {i}")
        break
    last = IS
