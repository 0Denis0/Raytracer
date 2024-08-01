from ray import Ray
from sphere import Sphere
import materials

class Hittable:
    def __init__(self) -> None:
        self.hittables = []

    def add(self, object):
        self.hittables.append(object)
    
    def clear(self):
            self.hittables = []

    def hit(self, ray):
        hitAnything = False
        objHit = -1
        tempClosest = 4096

        for i, obj in enumerate(self.hittables):
            currHit = obj.hit(ray)
            if(currHit > 0.00001):
                 hitAnything = True
                 if currHit < tempClosest:
                     tempClosest = currHit
                     objHit = i
        
        return hitAnything, (tempClosest, objHit)


    def rayColor(self, ray):
        pass

# class world1(Hittable):
#      def __init__(self) -> None:
#             super().__init__()
#             ball1 = Sphere(1,   [3, -0.75, 0], material=materials.Metal(albedo=[1, 1, 1]))
#             ball2 = Sphere(100, [3,  0,  101], material=materials.Lambertian(albedo=[0.7, 0.7, 0.7]))
#             ball3 = Sphere(1,   [4,  1,   -1], material=materials.Metal(albedo=[0.9, 0.9, 0]))
#             ball4 = Sphere(2,   [2, -3,   -1], material=materials.Lambertian(albedo=[0.9, 0.2, 0.9]))

#             self.add(ball1)
#             self.add(ball2)
#             self.add(ball3)
#             self.add(ball4)