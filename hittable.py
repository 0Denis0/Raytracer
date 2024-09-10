import numpy as np

from ray import Ray
from sphere import Sphere
import materials

class Hittable:
    def __init__(self) -> None:
        self.hittables = []

    def add(self, object):
        self.hittables.append(object)
    
    def clear(self):
            self.hittables = []

    def hit(self, rays):
        """
        Check for intersections between the world (all objects) and multiple rays using vectorized operations.

        Parameters:
        -----------
        rays : list or np.ndarray of Ray objects
            A list or array of rays, where each ray has an origin and direction.

        Returns:
        --------
        hitAnything : np.ndarray
            Boolean array indicating if each ray hit any object.
        hitData : tuple of (np.ndarray, np.ndarray)
            - Closest intersection distances for each ray.
            - Object indices of the closest object hit by each ray.
        """
        num_rays = len(rays)
        closest_dists = np.full(num_rays, 4096.0)  # Large value for initial closest distance
        obj_indices = np.full(num_rays, -1)  # -1 indicates no object hit

        # Iterate over all hittable objects
        for i, obj in enumerate(self.hittables):
            # Get intersection distances for the current object
            dists = obj.hit(rays)  # Vectorized hit check for all rays

            # Update closest hits
            hit_mask = (dists > 0.00001) & (dists < closest_dists)
            closest_dists[hit_mask] = dists[hit_mask]
            obj_indices[hit_mask] = i

        hitAnything = obj_indices != -1  # Boolean array: True if a ray hit something

        return hitAnything, (closest_dists, obj_indices)



    def rayColor(self, ray):
        pass

# class world1(Hittable):
#      def __init__(self) -> None:
#             super().__init__()
#             ball1 = Sphere(1,   [3, -0.75, 0], material=materials.Metal(albedo=[1, 1, 1]))
#             ball2 = Sphere(100, [3,  0,  101], material=materials.Lambertian(albedo=[0.7, 0.7, 0.7]))
#             ball3 = Sphere(1,   [4,  1,   -1], material=materials.Metal(albedo=[0.9, 0.9, 0]))
#             ball4 = Sphere(2,   [2, -3,   -1], material=materials.Lambertian(albedo=[0.9, 0.2, 0.9]))

#             self.add(ball1)
#             self.add(ball2)
#             self.add(ball3)
#             self.add(ball4)