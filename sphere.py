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
        # return np.array([-n[0], n[1], n[2]])
        # # return np.array([-n[2], n[1], -n[0]])
        # # return np.array([-n[2], n[1], n[0]])
        return (intersectionPt - self.center)/self.r
    
    def hit_old(self, ray):
        rel = self.center - ray.start
        # Quadratic eq coefficients:
        a = np.vdot(ray.vec, ray.vec)
        b = -2*np.vdot(ray.vec, rel)
        c = np.vdot(rel, rel) - self.r**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0.0001:
            # no hit
            return -1
        else:
            sqrt = math.sqrt(discriminant)
            return 0.5*(-b - sqrt)/a
        
    def hit(self, rays):
        """
        Check for intersections between the sphere and multiple rays using vectorized operations.

        Parameters:
        -----------
        rays : list or np.ndarray of Ray objects
            A list or array of rays, where each ray has an origin and direction.

        Returns:
        --------
        hits : np.ndarray
            Array of intersection distances for each ray. If a ray doesn't hit, the distance is -1.
        """
        # Extract ray origins and directions
        ray_starts = np.array([ray.start for ray in rays]).reshape(-1, 3)  # Shape: (num_rays, 3)
        ray_dirs = np.array([ray.vec for ray in rays]).reshape(-1, 3)  # Shape: (num_rays, 3)

        # Ensure that ray_starts and ray_dirs have the correct shape (num_rays, 3)
        assert ray_starts.ndim == 2 and ray_starts.shape[1] == 3, "ray_starts must have shape (num_rays, 3)"
        assert ray_dirs.ndim == 2 and ray_dirs.shape[1] == 3, "ray_dirs must have shape (num_rays, 3)"

        # Compute the vector from the ray origin to the sphere center
        rel = self.center - ray_starts  # Shape: (num_rays, 3)

        # Compute quadratic equation coefficients for all rays
        a = np.einsum('ij,ij->i', ray_dirs, ray_dirs)  # Dot product of each ray with itself, shape: (num_rays,)
        b = -2 * np.einsum('ij,ij->i', ray_dirs, rel)  # Dot product between ray direction and relative position
        c = np.einsum('ij,ij->i', rel, rel) - self.r**2  # Dot product of rel with itself minus sphere radius squared

        # Compute discriminant for the quadratic equation
        discriminant = b**2 - 4 * a * c

        # Initialize hit distances to -1 (for no hit)
        hits = np.full(len(rays), -1.0)

        # Only consider rays with positive discriminant (indicating a hit)
        mask = discriminant >= 0.0001

        # For rays that hit, calculate the intersection distances
        sqrt_discriminant = np.sqrt(discriminant[mask])
        hits[mask] = 0.5 * (-b[mask] - sqrt_discriminant) / a[mask]

        return hits


    
