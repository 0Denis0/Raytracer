from ray import Ray

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
            if(currHit > 0):
                 hitAnything = True
                 if currHit < tempClosest:
                     tempClosest = currHit
                     objHit = i
        
        return hitAnything, (tempClosest, objHit)


    def rayColor(self, ray):
        pass