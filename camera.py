import numpy as np
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
    def __init__(self, position = [0,0,0], angle = [0,0,0], 
                 vFOV = 90, aspectRatio = 16/9, imgWidth = 100, 
                 focalLen = 1, maxDepth = 10) -> None:
        self.pos = np.array(position)
        self.angle = np.array(angle)
        self.vFOV = vFOV
        self.vFOVRad = vFOV*math.pi/180
        self.focalLen = focalLen
        self.ar = aspectRatio
        self.imgWidth = int(imgWidth)
        self.imgHeight = int(imgWidth/aspectRatio)
        self.vMax = self.focalLen*math.tan(0.5*self.vFOVRad)
        self.vMin = -self.vMax
        # print("V span: " + str(self.vMax - self.vMin))
        self.uMax = aspectRatio*self.vMax
        self.uMin = -self.uMax
        # print("U span: " + str(self.uMax - self.uMin))
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

    def render(self, world, save=True):
        du = (self.uMax - self.uMin)/self.imgWidth
        dv = (self.vMax - self.vMin)/self.imgHeight
        for i in range(self.imgHeight):
            for j in range(self.imgWidth):
                ray = Ray(self.pos, [self.focalLen, self.uMin + du*j, self.vMin + dv*i])
                hit, tmp = world.hit(ray)
                if hit:
                    (dist, id) = tmp
                    ball = world.hittables[id]
                    # n = Ray(ray.start, ray.start + dist*ray.unit() - ball.center)
                    # n = ball.normal(ray.start + dist*ray.unit())
                    self.img[i, j] = np.sqrt(self.rayColor(ray, world, self.maxDepth))
                    # self.img[i, j] = 0.5 * (self.rayColor(ray, world, 1) + 1)
                    # self.img[i,j] = 0.5*(np.array([-n[1], n[2], n[0]]) + [1, 1, 1])
                else:
                    # self.img[i, j] = np.array([j/self.imgWidth, i/self.imgHeight, 1])
                    a = 0.5 * (-ray.unit()[2] + 1.0)
                    self.img[i,j] = np.sqrt(list((1.0 - a)*np.array([1, 1, 1]) + a * np.array([0.5, 0.7, 1.0])))
            print(str(self.imgHeight - i) + " rows left")

        print("Finished rendering.")
        if save:
            print("Saving...")
            self.saveImg()
            print("Saved.")
        # self.dispImg()

    # def renderParallel(self, world):
    #     du = (self.uMax - self.uMin)/self.imgWidth
    #     dv = (self.vMax - self.vMin)/self.imgHeight
    #     # for i in range(self.imgHeight):
    #     #     self.img[i] = self.renderLine(world, i, du, dv)
    #     with Pool() as pool:
    #         results = pool.map(self.renderLine, [(self.imgHeight+1)*[world], [i for i in range(self.imgHeight)], (self.imgHeight+1)*[du], (self.imgHeight+1)*[dv]])
    #     plt.imshow(results)
    #     plt.show()
    #     print("Finished rendering. Saving...")
    #     self.saveImg()
    #     print("Saved.")
    #     # self.dispImg()
    def render_line_helper(self, args):
        world, row, du, dv = args
        return self.renderLine(world, row, du, dv)

    def renderParallel(self, world, save=True):
        du = (self.uMax - self.uMin) / self.imgWidth
        dv = (self.vMax - self.vMin) / self.imgHeight
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
                ray = Ray(self.pos, [self.focalLen, self.uMin + du*j, self.vMin + dv*i])
                hit, tmp = world.hit(ray)
                if hit:
                    (dist, id) = tmp
                    ball = world.hittables[id]
                    imRow[0, j] = np.sqrt(self.rayColor(ray, world, self.maxDepth))
                else:
                    a = 0.5 * (-ray.unit()[2] + 1.0)
                    imRow[0, j] = np.sqrt(list((1.0 - a)*np.array([1, 1, 1]) + a * np.array([0.5, 0.7, 1.0])))
        # print(str(self.imgHeight - i) + " rows left")
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
            a = 0.5 * (np.ravel(ray.unit())[2] + 1.0)
            return (1.0 - a)*np.array([1, 1, 1]) + a * np.array([0.5, 0.7, 1.0])