import numpy as np
import matplotlib.pyplot as plt
import time
import PIL
import datetime

from camera import Camera
from ray import Ray
from sphere import Sphere
from hittable import Hittable
import materials

def main():
    world = Hittable()
    ball1 = Sphere(1,   [3, -0.75, 0], material=materials.Metal(albedo=[1, 1, 1]))
    ball2 = Sphere(100, [3,  0,  101], material=materials.Lambertian(albedo=[0.7, 0.7, 0.7]))
    ball3 = Sphere(1,   [4,  1,   -1], material=materials.Metal(albedo=[0.9, 0.9, 0]))
    ball4 = Sphere(2,   [2, -3,   -1], material=materials.Lambertian(albedo=[0.9, 0.2, 0.9]))

    # ball1 = Sphere(100, [105, 0, 95], material=materials.Material(albedo=[0.2, 0.2, 0.2]))
    # ball2 = Sphere(100, [3, 0, 101], material=materials.Material(albedo=[0, 0, 0.2]))
    # ball3 = Sphere(100, [3, 105, 0], material=materials.Material(albedo=[0, 0.2, 0]))
    # ball4 = Sphere(100, [3, -105, 0], material=materials.Material(albedo=[0.9, 0, 0]))
    # ball5 = Sphere(1, [3, 0, 0], material=materials.Material(albedo=[5, 5, 5]))

    world.add(ball1)
    world.add(ball2)
    world.add(ball3)
    world.add(ball4)
    # world.add(ball5)

    # ray1 = Ray(cam1.pos, ball1.center)
    # ball1.hit(ray1)
    dataPts = 100
    results = np.zeros((dataPts, 3))
    for i in range(dataPts):
        width = i*16 + 16
        cam1 = Camera(maxDepth=10, imgWidth=width)
        cam1.imgWidth = width

        start = time.time()
        cam1.render(world, save=False)
        end = time.time()
        classic_t = end - start

        start = time.time()
        cam1.renderParallel(world, save=False)
        end = time.time()
        parallel_t = end - start

        results[i] = [width, classic_t, parallel_t]

    plt.plot(results[:, 0], results[:, 1], label='Serial Compute')
    plt.plot(results[:, 0], results[:, 2], label='Parallel Compute')
    plt.xlabel("Image Width [px]")
    plt.ylabel("Time To Render [s]")
    plt.legend()
    plt.show()
    # print(cam1.img[int(cam1.imgHeight/2), :])

if __name__ == '__main__':
    main()
