import numpy as np
import matplotlib.pyplot as plt
import PIL
import datetime

from camera import Camera
from ray import Ray
from sphere import Sphere
from hittable import Hittable

cam1 = Camera()

world = Hittable()
ball1 = Sphere(1, [5, 0, 0])
ball2 = Sphere(100, [5, 0, 101])
ball3 = Sphere(1, [3, 1, -1])

world.add(ball1)
world.add(ball2)
world.add(ball3)

ray1 = Ray(cam1.pos, ball1.center)
ball1.hit(ray1)

cam1.render(world)
# print(cam1.img[int(cam1.imgHeight/2), :])
