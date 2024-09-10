import numpy as np
from numpy.linalg import norm as mag
import matplotlib.pyplot as plt
import math
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm
import os

from ray import Ray
from hittable import Hittable
import materials

class Camera:
    '''
    Camera class for raytracer

    Attributes:
    -----------
    pos : 1x3 Array
        Camera Position as [x, y, z].

    angle : 1x3 Array
        Camera angle/attitude as [roll, pitch, yaw].

    vFOV : float
        Vertical field of view in degrees.

    aspectRatio : float
        Image aspect ratio width/height


    Methods:
    ----------

    TBD
    '''

    def __init__(self, position = [0,0,0], lookAt = [1,0,0], vUp = [0, 0, -1],
                 vFOV = 90, aspectRatio = 16/9, imgWidth = 100, 
                 maxDepth = 10, raysPerPixel = 1) -> None:
        self.position = np.array(position)
        self.lookAt = np.array(lookAt)
        self.vUp = np.array(vUp)
        self.vFOV = vFOV
        self.aspectRatio = aspectRatio
        self.imgWidth = imgWidth
        self.maxDepth = maxDepth
        self.raysPerPixel = raysPerPixel
        self.recalculate()

    def recalculate(self):
        self.lookVec = self.lookAt - self.position
        self.vFOVRad = self.vFOV * math.pi / 180
        self.focalLen = np.linalg.norm(self.lookVec)
        self.imgHeight = int(self.imgWidth / self.aspectRatio)
        self.viewportHeight = 2 * self.focalLen * math.tan(0.5 * self.vFOVRad)
        self.viewportWidth = self.viewportHeight * (self.imgWidth / self.imgHeight)
        
        self.w = -self.lookVec / mag(self.lookVec)
        self.u = np.cross(self.vUp, self.w) / mag(np.cross(self.vUp, self.w))
        self.v = np.cross(self.w, self.u) / mag(np.cross(self.w, self.u))
        self.viewU = self.u * self.viewportWidth
        self.viewV = -self.v * self.viewportHeight
        
        self.img = np.zeros((self.imgHeight, self.imgWidth, 3))

    def updateVars(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, np.array(value) if isinstance(value, list) else value)
        self.recalculate()

    def dispImg(self):
        plt.imshow(self.img)
        plt.axis('off')
        plt.show()

    # def saveImg(self, dir="renders\\", name=""):
    #     print("Saving...")
    #     if name == "":
    #         now = datetime.datetime.now()
    #         t = now.strftime("%Y%m%d_%H%M%S")
    #         name = "test" + t + ".png"
    #     plt.imshow(self.img)
    #     plt.axis('off')
    #     plt.savefig(dir + name, bbox_inches='tight')
    #     print("Saved.")

    def saveImg(self, dir="renders\\", name=""):
        """
        Save the rendered image in full resolution.

        Parameters:
        -----------
        dir : str
            Directory where the image will be saved. Default is 'renders\\'.
        name : str
            Name of the image file. If empty, a timestamped name is generated.
        """
        print("Saving...")

        # Generate default file name if not provided
        if name == "":
            now = datetime.datetime.now()
            t = now.strftime("%Y%m%d_%H%M%S")
            name = "render_" + t + ".png"
        
        # Ensure the directory exists
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        # Get the image dimensions (height, width)
        height, width, _ = self.img.shape

        # Create a new figure with the correct size in inches (width / height)
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        
        # Display the image without axis
        ax.imshow(self.img)
        ax.axis('off')

        # Save the figure in full resolution by setting bbox_inches='tight'
        plt.savefig(os.path.join(dir, name), bbox_inches='tight', pad_inches=0)

        # Close the figure to free up memory
        plt.close(fig)

        print(f"Saved image as: {os.path.join(dir, name)}")


    def renderParallel(self, world, save=True):
        du = self.viewU/self.imgWidth
        dv = self.viewV/self.imgHeight
        args_list = [(world, i, du, dv, self.raysPerPixel) for i in range(self.imgHeight)]
        
        pool = Pool()
        results = []
        for result in tqdm.tqdm(pool.imap(self.render_line_helper, args_list), total=len(args_list)):
            results.append(result)
        pool.close()
        pool.join()

        # Combine results into the image array
        for i, row in enumerate(results):
            self.img[i] = row
        
        print("Finished rendering.")
        if save:
            self.saveImg()

    def render_line_helper(self, args):
        world, row, du, dv, raysPerPixel = args
        return self.renderLine(world, row, du, dv, raysPerPixel)

    def renderLine(self, world, row, du, dv, raysPerPixel):
        """
        Render a single row of the image using a vectorized approach with NumPy.

        Parameters:
        -----------
        world : World object
            The scene or world containing objects and lights.
        row : int
            The row of the image to render.
        du : np.ndarray
            The horizontal pixel size in world coordinates as a 3D vector.
        dv : np.ndarray
            The vertical pixel size in world coordinates as a 3D vector.
        raysPerPixel : int
            The number of rays to shoot per pixel for anti-aliasing.

        Returns:
        --------
        imRow : np.ndarray
            The rendered row as a 1 x width x 3 NumPy array of RGB values.
        """
        i = row  # The current row being processed
        imRow = np.zeros((1, self.imgWidth, 3))  # Initialize output row with zeros

        # Generate pixel indices for the entire row (j values for each pixel in this row)
        j_values = np.arange(self.imgWidth)
        
        # Generate random jitter for each pixel and each ray (to apply anti-aliasing)
        jitter_j = j_values[:, None] + np.random.random((self.imgWidth, raysPerPixel))  # Horizontal jitter
        jitter_i = i + np.random.random((self.imgWidth, raysPerPixel))  # Vertical jitter
        
        # Reshape jitter arrays to match the direction vector's shape (3, 1920, 255)
        jitter_j = jitter_j[np.newaxis, :, :]  # (1, 1920, 255) -> add a new axis for broadcasting
        jitter_i = jitter_i[np.newaxis, :, :]  # (1, 1920, 255)

        # Reshape du and dv to match the jitter array shape (3, 1, 1) so they can broadcast correctly
        du = du[:, np.newaxis, np.newaxis]  # Shape becomes (3, 1, 1)
        dv = dv[:, np.newaxis, np.newaxis]  # Shape becomes (3, 1, 1)

        # Compute ray directions for all rays in the row, accounting for jitter and offsets (du and dv)
        ray_directions = (
            -self.focalLen * self.w[:, np.newaxis, np.newaxis]  # Camera direction (w) expanded
            - 0.5 * self.viewU[:, np.newaxis, np.newaxis]  # Half offset in the U direction
            + du * jitter_j  # Apply horizontal jitter using du
            - 0.5 * self.viewV[:, np.newaxis, np.newaxis]  # Half offset in the V direction
            + dv * jitter_i  # Apply vertical jitter using dv
        )
        
        # Normalize ray directions to unit length
        ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=0, keepdims=True)

        # Create Ray objects (vectorized) - Same origin for all rays, but different directions
        ray_origins = np.tile(self.position, (self.imgWidth * raysPerPixel, 1))  # Replicate camera position for all rays
        ray_directions = ray_directions.reshape(3, -1).T  # Flatten ray directions to (total_rays, 3)

        # Create a list of Ray objects (each with an origin and a direction)
        rays = [Ray(ray_origins[i], ray_directions[i]) for i in range(len(ray_origins))]

        # Call the vectorized rayColor function with all the rays
        colors = self.rayColor(rays, world, self.maxDepth)

        # Reshape the colors back to (imgWidth, raysPerPixel, 3)
        colors = colors.reshape(self.imgWidth, raysPerPixel, 3)

        # Average over the raysPerPixel to get the final color for each pixel
        imRow[0] = np.clip(np.sqrt(np.mean(colors, axis=1)), 0, 1)  # Apply gamma correction (sqrt) and clip the values

        return imRow



    def renderSimple(self, world):
        du = self.viewU/self.imgWidth
        dv = self.viewV/self.imgHeight
        for i in range(self.imgHeight):
            for j in range(self.imgWidth):
                ray = Ray(self.position, 
                          -self.focalLen * self.w 
                          - 0.5 * self.viewU + du*j
                           - 0.5 * self.viewV + dv*i)
                self.img[i, j] = (self.rayColor(ray, world, self.maxDepth))
                # np.sqrt for linear space. 

            print(f"Rendered row {i}")
        self.saveImg()
        
    def rayColor(self, rays, world, depth):
        """
        Vectorized version of rayColor to handle multiple rays in parallel.

        Parameters:
        -----------
        rays : np.ndarray
            Array of rays to be processed. Each ray is represented by its origin and direction.
        world : World
            The scene or world containing objects and lights.
        depth : int
            Maximum recursion depth for reflections.

        Returns:
        --------
        colors : np.ndarray
            Array of colors computed for each input ray.
        """
        num_rays = len(rays)
        colors = np.zeros((num_rays, 3))  # Initialize color array for each ray
        
        active_rays = np.ones(num_rays, dtype=bool)  # Mask for active rays
        remaining_depth = np.full(num_rays, depth)   # Track depth for each ray
        
        for d in range(depth):
            if not active_rays.any():
                break
            
            # Process rays that are still active
            active_rays_indices = np.where(active_rays)[0]
            active_rays_batch = [rays[i] for i in active_rays_indices]
            
            # Ray-object intersections for the active rays
            hitSmth, hitData = world.hit(active_rays_batch)
            
            # Split hit data
            hit_dists, hit_ids = hitData
            
            # Compute hit points for rays that hit objects
            hitPts = np.array([rays[i].at(hit_dists[i]) for i in active_rays_indices])
            
            # Fetch objects and normals at hit points
            objs = [world.hittables[hit_ids[i]] for i in active_rays_indices]
            normals = np.array([objs[i].normal(hitPts[i]) for i in range(len(objs))])
            mats = np.array([objs[i].material for i in range(len(objs))])
            
            # Update colors based on hit object material
            albedo_values = np.array([mat.albedo for mat in mats])
            albedo_mask = np.any(albedo_values > 1, axis=1)
            colors[active_rays_indices[albedo_mask]] = albedo_values[albedo_mask]
            
            # Compute reflection rays for non-terminal hits
            reflect_dirs = np.array([mat.reflect(rays[i], normals[i]) for i, mat in enumerate(mats)])
            new_rays = np.array([Ray(hitPts[i], reflect_dirs[i]) for i in range(len(hitPts))])
            
            # Update rays and depth for the next iteration
            rays = new_rays
            remaining_depth -= 1
            
            # Handle termination condition for depth
            active_rays[remaining_depth <= 0] = False
        
        # For rays that didn't hit anything, use the background color
        no_hit_mask = np.logical_not(hitSmth)
        background_rays = rays[no_hit_mask]
        
        if background_rays.any():
            ray_dirs = np.array([ray.unit() for ray in background_rays]).reshape(-1, 3)
            a = 0.5 * (-ray_dirs[:, 2] + 1.0)  # Background gradient factor
            background_colors = (1.0 - a[:, None]) * np.array([1, 1, 1]) + a[:, None] * np.array([0.5, 0.7, 1.0])
            colors[no_hit_mask] = background_colors
        
        return colors

        