import numpy as np

class Ray:
    '''
    Class for defining a ray in 3D space and its properties

    Attributes:
    -----------
    start : list
        Source point of the ray, [x0, y0, z0]
    
    end : list
        End point of the ray, [x_final, y_final, z_final]

    Methods:
    ----------
    unit() : returns unit vector in the same direction as ray

    mag() : returns the magnitude of the ray
    '''
    
    def __init__(self, start, end) -> None:
        self.start = np.array(start)
        self.end = np.array(end)
        self.vec = self.end - self.start

    def unit(self) -> np.array:
        return self.vec * (1/self.mag())

    def mag(self) -> int:
        return np.linalg.norm(self.vec)
    
    
