# Testing

import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((1,1,3))
img[0,0] = np.array([0.20818288, 0.34297366, 0.10568163])
plt.imshow(img)
plt.show()