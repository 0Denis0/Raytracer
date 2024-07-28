import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

from ray import Ray
from hittable import Hittable

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
    def __init__(self, position = [0,0,0], angle = [0,0,0], vFOV = 90, aspectRatio = 16/9, imgWidth = 100, focalLen = 1) -> None:
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

        self.img = np.zeros((self.imgHeight, self.imgWidth, 3))


    def dispImg(self):
        plt.imshow(self.img)
        plt.axis('off')
        plt.show()

    def saveImg(self):
        now = datetime.datetime.now()
        t = now.strftime("%Y%m%d_%H%M%S")
        plt.imshow(self.img)
        plt.axis('off')
        plt.savefig("renders\\test" + t + ".png", bbox_inches='tight')

    def render(self, world):
        du = (self.uMax - self.uMin)/self.imgWidth
        dv = (self.vMax - self.vMin)/self.imgHeight
        for i in range(self.imgHeight):
            for j in range(self.imgWidth):
                ray = Ray(self.pos, [self.focalLen, self.uMin + du*j, self.vMin + dv*i])
                hit, tmp = world.hit(ray)
                if hit:
                    (dist, id) = tmp
                    ball = world.hittables[id]
                    n = Ray(ray.start, ray.start + dist*ray.unit() - ball.center)
                    self.img[i, j] = 0.5 * (n.vec + [1, 1, 1])
                else:
                    self.img[i, j] = np.array([j/self.imgWidth, i/self.imgHeight, 1])
            print(str(self.imgHeight - i) + " rows left")

        print("Finished rendering. Saving...")
        self.dispImg()
        self.saveImg()
        print("Saved.")

