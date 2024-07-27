import numpy as np
import math

from ray import Ray

class Sphere:
    def __init__(self, radius, center) -> None:
        self.r = radius
        self.center = np.array(center)

    def normal(self, intersectionPt):
        return intersectionPt - self.center
    
    def hit(self, ray):
        rel = self.center - ray.start
        # Quadratic eq coefficients:
        a = np.vdot(ray.vec, ray.vec)
        b = -2*np.vdot(ray.vec, rel)
        c = np.vdot(rel, rel) - self.r**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            # no hit
            return -1
        else:
            return (-b - math.sqrt(b**2 - 4*a*c))/(2*a)
        

    
