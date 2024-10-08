import numpy as np
import matplotlib.pyplot as plt
import time
import os

from camera import Camera
from sphere import Sphere
from hittable import Hittable
import materials
from genVideo import genVideo, pngs_to_gif

def main():
    world = Hittable()
    ball1 = Sphere(1,   [3, -0.75, 0], material=materials.Metal(albedo=[0.5, 0.5, 0.5]))
    ball2 = Sphere(1000, [3,  0,  1001], material=materials.Lambertian(albedo=[0.7, 0.7, 0.7]))
    ball3 = Sphere(1,   [5,  1,   -1], material=materials.Metal(roughness=0, albedo=[0.9, 0.9, 0]))
    ball4 = Sphere(2,   [2, -3,   -1], material=materials.Material(albedo=[0.9, 0.2, 0.9]))
    ball5 = Sphere(0.25, [2.5, -1.5, 0.75], material=materials.Lambertian(albedo=[5, 5, 5]))
    # ball1 = Sphere(100, [105, 0, 95], material=materials.Material(albedo=[0.9, 0.9, 0.9]))
    # ball2 = Sphere(100, [3, 0, 101], material=materials.Material(albedo=[0, 0, 0.9]))
    # ball3 = Sphere(100, [3, 105, 0], material=materials.Material(albedo=[0, 0.9, 0]))
    # ball4 = Sphere(100, [3, -105, 0], material=materials.Material(albedo=[0.9, 0, 0]))
    # ball5 = Sphere(1, [3, 0, 0], material=materials.Material(albedo=[50, 50, 50]))


    world.add(ball1)
    world.add(ball2)
    world.add(ball3)
    world.add(ball4)
    world.add(ball5)

    cam1 = Camera(maxDepth=800, imgWidth=1920, vFOV=90, raysPerPixel=255)
    # # cam1.renderSimple(world)
    # # cam1.dispImg()
    # folder = "renders/vid5/"
    # start = time.time()

    # frames = 60
    # speed = 2*np.pi/frames

    # for i in range(frames):
    #     frameNum = f"{i:04d}"
    #     fileName = os.path.join(folder, f"frame{frameNum}.png")
    #     if not os.path.exists(fileName):
    #         print(f"Rendering Frame {frameNum}")
    #         cam1.updateVars(position=cam1.lookAt + [-8*np.cos(i*speed), -8*np.sin(i*speed), -2])
    #         cam1.renderParallel(world, save=False)
    #         cam1.saveImg(folder, name=f"frame{frameNum}.png")
    #     else:
    #         print(f"Frame {frameNum} already exists. Skipping...")
    
    # end = time.time()
    # parallel_t = end - start
    # print("Frame render time:", parallel_t)
    # # cam1.dispImg()

    # # genVideo("renders/vid5", "renders/vid5/motion5.gif")
    
    # png_folder_path = 'renders/vid5'
    # output_gif_path = 'renders/vid5/motionFromPngs.gif'
    # pngs_to_gif(png_folder_path, output_gif_path, fps=30)

    start = time.time()
    cam1.renderParallel(world)
    tot = time.time() - start
    print(f"Total render time: {tot}.")
    # cam1.dispImg()

if __name__ == '__main__':
    main()
