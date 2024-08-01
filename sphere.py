import numpy as np
import math

from ray import Ray
import materials

class Sphere:
    def __init__(self, radius, center, material = materials.Material()) -> None:
        self.r = radius
        self.center = np.array(center)
        self.material = material

    def setMaterial(self, material):
        self.material = material

    def normal(self, intersectionPt):
        # n = np.ravel((intersectionPt - self.center)/self.r)
        # return np.array([n[2], -n[1], -n[0]])
        # # return np.array([-n[2], n[1], -n[0]])
        # # return np.array([-n[2], n[1], n[0]])
        return (intersectionPt - self.center)/self.r
    
    def hit(self, ray):
        rel = self.center - ray.start
        # Quadratic eq coefficients:
        a = np.vdot(ray.vec, ray.vec)
        b = -2*np.vdot(ray.vec, rel)
        c = np.vdot(rel, rel) - self.r**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0.00001:
            # no hit
            return -1
        else:
            sqrt = math.sqrt(discriminant)
            return 0.5*(-b - sqrt)/a
        

    
