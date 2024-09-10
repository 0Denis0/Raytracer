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
        i = row
        imRow = np.zeros((1, self.imgWidth, 3))
        for j in range(self.imgWidth):
            for _ in range(raysPerPixel):
                ray = Ray(self.position, -self.focalLen * self.w 
                        - 0.5 * self.viewU + du*(j + np.random.random())
                        - 0.5 * self.viewV + dv*(i + np.random.random()))
                imRow[0, j] += np.clip(np.sqrt(self.rayColor(ray, world, self.maxDepth)), 0, 1)
        imRow = imRow/raysPerPixel
        # print(f"Rendered row {i}.")
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
        