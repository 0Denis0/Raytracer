import numpy as np
from numpy.linalg import norm as mag
import matplotlib.pyplot as plt
import math
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

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
                 maxDepth = 10) -> None:
        self.pos = np.array(position)
        self.lookAt = np.array(lookAt)
        self.lookVec = self.lookAt - self.pos
        self.vFOV = vFOV
        self.vFOVRad = vFOV*math.pi/180
        self.focalLen = np.linalg.norm(self.lookVec)
        self.ar = aspectRatio
        self.imgWidth = int(imgWidth)
        self.imgHeight = int(imgWidth/aspectRatio)
        self.viewportHeight = 2 * self.focalLen * math.tan(0.5 * self.vFOVRad)
        self.viewportWidth = self.viewportHeight * (self.imgWidth/self.imgHeight)
        
        self.w = -self.lookVec/mag(self.lookVec)
        self.u = np.cross(vUp, self.w)/mag(np.cross(vUp, self.w))
        self.v = np.cross(self.w, self.u)/mag(np.cross(self.w, self.u))
        self.viewU = self.u * self.viewportWidth
        self.viewV = -self.v * self.viewportHeight
        
        self.maxDepth = maxDepth

        self.img = np.zeros((self.imgHeight, self.imgWidth, 3))

    def dispImg(self):
        plt.imshow(self.img)
        plt.axis('off')
        plt.show()

    def saveImg(self):
        now = datetime.datetime.now()
        t = now.strftime("%Y%m%d_%H%M%S")
        plt.imshow(self.img, interpolation='quadric')
        plt.axis('off')
        plt.savefig("renders\\test" + t + ".png", bbox_inches='tight')

    def render_line_helper(self, args):
        world, row, du, dv = args
        return self.renderLine(world, row, du, dv)

    def renderParallel(self, world, save=True):
        du = self.viewU/self.imgWidth
        dv = self.viewV/self.imgHeight
        args_list = [(world, i, du, dv) for i in range(self.imgHeight)]
        
        with Pool() as pool:
            results = pool.map(self.render_line_helper, args_list)
        
        # Combine results into the image array
        for i, result in enumerate(results):
            self.img[i] = result
        
        print("Finished rendering.")
        if save:
            print("Saving...")
            self.saveImg()
            print("Saved.")


    def renderLine(self, world, row, du, dv):
        i = row
        imRow = np.zeros((1, self.imgWidth, 3))
        for j in range(self.imgWidth):
                ray = Ray(self.pos, -self.focalLen * self.w 
                          - 0.5 * self.viewU + du*j
                           - 0.5 * self.viewV + dv*i)
                imRow[0, j] = np.sqrt(self.rayColor(ray, world, self.maxDepth))
        print(f"Rendered row {i}.")
        return imRow

    def rayColor(self, ray, world, depth):
        if depth <= 0:
            return [0, 0, 0]
        
        hitSmth, hitData = world.hit(ray)
        if hitSmth:
            dist, id = hitData
            hitPt = ray.start + dist*ray.unit()
            obj = world.hittables[id]
            n = obj.normal(hitPt)
            mat = obj.material
            if (mat.albedo > 1).any():
                return mat.albedo
            newRay = Ray(hitPt, mat.reflect(ray, n))
            return mat.albedo * np.array(self.rayColor(newRay, world, depth-1))
        else:
            # return np.array([0,0,0])
            a = 0.5 * (np.ravel(-ray.unit())[2] + 1.0)
            return (1.0 - a)*np.array([1, 1, 1]) + a * np.array([0.5, 0.7, 1.0])