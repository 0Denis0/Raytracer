import numpy as np
import matplotlib.pyplot as plt
import PIL
import datetime

from camera import Camera
from ray import Ray
from sphere import Sphere
from hittable import Hittable
import materials

cam1 = Camera(maxDepth=100, imgWidth=240)

world = Hittable()
# ball1 = Sphere(1, [3, -0.75, 0], material=materials.Lambertian(albedo=[1, 1, 1]))
# ball2 = Sphere(100, [3, 0, 101], material=materials.Lambertian(albedo=[0.7, 0.7, 0.7]))
# ball3 = Sphere(1, [4, 1, -1], material=materials.Lambertian(albedo=[0.9, 0.9, 0]))
# ball4 = Sphere(2, [2, -3, -1], material=materials.Lambertian(albedo=[0.9, 0.2, 0.9]))

ball1 = Sphere(100, [105, 0, 95], material=materials.Lambertian(albedo=[0.9, 0.9, 0.9]))
ball2 = Sphere(100, [3, 0, 101], material=materials.Lambertian(albedo=[0, 0, 0.9]))
ball3 = Sphere(100, [3, 105, 0], material=materials.Lambertian(albedo=[0, 0.9, 0]))
ball4 = Sphere(100, [3, -105, 0], material=materials.Lambertian(albedo=[0.9, 0, 0]))

world.add(ball1)
world.add(ball2)
world.add(ball3)
world.add(ball4)

# ray1 = Ray(cam1.pos, ball1.center)
# ball1.hit(ray1)

cam1.render(world)
# print(cam1.img[int(cam1.imgHeight/2), :])
