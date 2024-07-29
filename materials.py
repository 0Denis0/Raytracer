# materials 

import numpy as np

from ray import Ray

class Material:
    def __init__(self, albedo=[0.5, 0.5, 0.5]) -> None:
        self.albedo = np.array(albedo)

    def reflect(self, ray, normal):
        # random diffuse scatter
        return ray.randOnHemisphere(normal)
    
class Metal(Material):
    def reflect(self, ray, normal):
        # mirror reflection
        return ray.vec - 2*normal*np.vdot(ray.vec, normal)
    
class Lambertian(Material):
    def reflect(self, ray, normal):
        # lambertian diffuse scatter
        return ray.randOnHemisphere(normal) + normal