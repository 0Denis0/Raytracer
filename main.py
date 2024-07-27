import numpy as np
import matplotlib.pyplot as plt
import PIL
import datetime

from camera import Camera
from ray import Ray
from sphere import Sphere
from hittable import Hittable

cam1 = Camera(imgWidth=480)

world = Hittable()
ball1 = Sphere(1, [5, 0, 0])

ray1 = Ray(cam1.pos, ball1.center)
ball1.hit(ray1)

cam1.render(ball1)
print(cam1.img[int(cam1.imgHeight/2), :])
