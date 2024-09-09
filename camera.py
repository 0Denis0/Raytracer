import numpy as np
from numpy.linalg import norm as mag
import matplotlib.pyplot as plt
import math
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm

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

    def saveImg(self, dir="renders\\", name=""):
        print("Saving...")
        if name == "":
            now = datetime.datetime.now()
            t = now.strftime("%Y%m%d_%H%M%S")
            name = "test" + t + ".png"
        plt.imshow(self.img)
        plt.axis('off')
        plt.savefig(dir + name, bbox_inches='tight')
        print("Saved.")

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
        du : float
            The horizontal pixel size in world coordinates.
        dv : float
            The vertical pixel size in world coordinates.
        raysPerPixel : int
            The number of rays to shoot per pixel for anti-aliasing.

        Returns:
        --------
        imRow : np.ndarray
            The rendered row as a 1 x width x 3 NumPy array of RGB values.
        """
        i = row
        imRow = np.zeros((1, self.imgWidth, 3))
        
        # Generate j values for each pixel in the row
        j_values = np.arange(self.imgWidth)
        
        # Generate random samples for anti-aliasing (raysPerPixel samples per pixel)
        jitter_j = j_values[:, None] + np.random.random((self.imgWidth, raysPerPixel))
        jitter_i = i + np.random.random((self.imgWidth, raysPerPixel))
        
        # Compute ray directions for the entire row at once
        ray_directions = (-self.focalLen * self.w 
                        - 0.5 * self.viewU + du * jitter_j 
                        - 0.5 * self.viewV + dv * jitter_i)
        
        # Normalize ray directions
        ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)
        
        # Create an array of rays for each pixel
        rays = [Ray(self.position, direction) for direction in ray_directions.reshape(-1, 3)]
        
        # Compute ray colors for all rays in a vectorized way
        colors = np.array([self.rayColor(ray, world, self.maxDepth) for ray in rays])
        
        # Reshape colors back to (imgWidth, raysPerPixel, 3) and average over raysPerPixel
        colors = colors.reshape(self.imgWidth, raysPerPixel, 3)
        
        # Average colors over the raysPerPixel
        imRow[0] = np.clip(np.sqrt(np.mean(colors, axis=1)), 0, 1)

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
        
    def rayColor(self, ray, world, depth):
        if depth <= 0:
            return [0, 0, 0]
        
        hitSmth, hitData = world.hit(ray)
        if hitSmth:
            dist, id = hitData
            hitPt = ray.at(dist)
            obj = world.hittables[id]
            n = obj.normal(hitPt)
            mat = obj.material
            if (mat.albedo > 1).any():
                return mat.albedo
            newRay = Ray(hitPt, hitPt + mat.reflect(ray, n))
            return mat.albedo * np.array(self.rayColor(newRay, world, depth-1))
        else:
            # return np.array([0,0,0])
            a = 0.5 * (-np.ravel(ray.unit())[2] + 1.0)
            return (1.0 - a)*np.array([1, 1, 1]) + a * np.array([0.5, 0.7, 1.0])
        