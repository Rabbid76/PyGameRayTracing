# Raytrace renderer
# 
# Implemented for PyGame [https://www.pygame.org]
#
# Based on:
#  
# "Ray Tracing: the Next Week (Ray Tracing Minibooks Book 2)"
# ASIN: B01B5AODD8
# [https://www.amazon.com/Ray-Tracing-Weekend-Minibooks-Book-ebook/dp/B01B5AODD8]


import pygame
import ctypes
import math
import random
import threading 
import os
import sys

currentWDir = os.getcwd()
print( 'current working directory: {}'.format( str(currentWDir) ) )
fileDir = os.path.dirname(os.path.abspath(__file__)) # det the directory of this file
print( 'current location of self: {}'.format( str(fileDir) ) )
parentDir = os.path.abspath(os.path.join(fileDir, os.pardir)) # get the parent directory of this file
sys.path.insert(0, parentDir)
print( 'insert system directory: {}'.format( str(parentDir) ) )
os.chdir( fileDir )
baseWDir = os.getcwd()
print( 'changed current working directory: {}'.format( str(baseWDir) ) )
print ( '' )

# globals
BLACK = (0, 0, 0)
RED = (255, 0, 0)

vec3 = pygame.Vector3
rand01 = random.random

###################################################################################################
# write PPM file
def writeFilePPM(surface, name):
    try:
        nx, ny = surface.get_size()
        file = open(name + '.ppm', "w")
    except:
        return 
    file.write("P3\n" + str(nx) + " " + str(ny) + "\n255\n")
    for y in range(ny):
        for x in range(nx):
            c = surface.get_at((x, y))
            file.write(str(c[0]) + " " + str(c[1]) + " " + str(c[2]) + " ")
        file.write("\n")
    file.close() 


###################################################################################################
# color conversions
def fToColor(c):
    # gc = c; 
    gc = (math.sqrt(c[0]), math.sqrt(c[1]), math.sqrt(c[2])) # gamma 2
    return (int(gc[0]*255.5), int(gc[1]*255.5), int(gc[2]*255.5))
def surfaceSetXYWHi(s, x, y, w, h, c):
    if c[0] > 255 or c[1] > 255 or c[2] > 255:
        c = (max(0, min(255, c[0])), max(0, min(255, c[1])), max(0, min(255, c[2])))
    if w > 1 or h > 1: 
        s.fill(c, (x, s.get_height()-y-h, w, h))
    else:
        s.set_at((x, s.get_height()-y-1), c)
def surfaceSetXYWHf(s, x, y, w, h, c):
    surfaceSetXYWHi(s, x, y, w, h, fToColor(c))


###################################################################################################
# vector generation and operation functions
def multiply_components(a, b):
    return vec3(a[0]*b[0],a[1]*b[1],a[2]*b[2])
def random_in_unit_sphere(): 
    while True: 
        # random vector x, y, z in [-1, 1]
        p = 2*vec3(rand01(), rand01(), rand01()) - vec3(1, 1, 1)  
        if p.magnitude_squared() < 1: # magnitude of vector has to be less than 1 
            break
    return p
def random_in_unit_disk(): 
    while True: 
        # random vector x, y, z in [-1, 1]
        p = 2*vec3(rand01(), rand01(), 0) - vec3(1, 1, 0)  
        if p.magnitude_squared() < 1: # magnitude of vector has to be less than 1 
            break
    return p
def reflect(v, n):
    return v - 2*v.dot(n)*n
def refract(v, n, ni_over_nt):
    # Snell's law: n*sin(theta) = n'*sin(theta')
    uv = v.normalize()
    dt = uv.dot(n)
    discriminant = 1 - ni_over_nt*ni_over_nt*(1-dt*dt)
    if discriminant > 0:
        return ni_over_nt*(uv-n*dt) - n*math.sqrt(discriminant)
    return None
def schlick(cosine, ref_idx):
    r0 = (1-ref_idx) / (1+ref_idx)
    r0 = r0*r0
    return r0 + (1-r0)*math.pow(1-cosine, 5)


###################################################################################################
# Main application window and process
class Application:

    #----------------------------------------------------------------------------------------------
    # ctor
    def __init__(self, size = (800, 600), caption = "PyGame window"):

        # state attributes
        self.__run = True

        # PyGame initialization
        pygame.init()
        self.__init_surface(size)
        pygame.display.set_caption(caption)
        
        self.__clock = pygame.time.Clock()

    #----------------------------------------------------------------------------------------------
    # dtor
    def __del__(self):
        pygame.quit()

    #----------------------------------------------------------------------------------------------
    # set the size of the application window
    @property
    def size(self):
        return self.__surface.get_size()

    #----------------------------------------------------------------------------------------------
    # get window surface
    @property
    def surface(self):
        return self.__surface

    #----------------------------------------------------------------------------------------------
    # get and set application 
    @property
    def image(self):
        return self.__image
    @image.setter
    def image(self, image):
        self.__image = image

    #----------------------------------------------------------------------------------------------
    # main loop of the application 
    def run(self, render, samples_per_pixel = 100, samples_update_rate = 1, capture_interval_s = 0):
        size = self.__surface.get_size()
        render.start(size, samples_per_pixel, samples_update_rate)
        finished = False
        start_time = None
        capture_i = 0
        while self.__run:
            self.__clock.tick(60)
            self.__handle_events()
            current_time = pygame.time.get_ticks()
            if start_time == None:
                start_time = current_time + 1000
            if not self.__run:
                render.stop()
            elif size != self.__surface.get_size():
                size = self.__surface.get_size()
                render.stop()
                render.start(size, samples_per_pixel, samples_update_rate)   
            capture_frame = capture_interval_s > 0 and current_time >= start_time + capture_i * capture_interval_s * 1000
            frame_img = self.draw(render, capture_frame)
            if frame_img:
                pygame.image.save(frame_img, "capture/img_" + str(capture_i) + ".png")
                capture_i += 1
            if not finished and not render.in_progress():
                finished = True
                print("Render time:", (current_time-start_time)/1000, " seconds")

        self.__render_image = render.copy()
        writeFilePPM(self.__render_image, "rt_2")
        pygame.image.save(self.__render_image, "rt_2.png")

    #----------------------------------------------------------------------------------------------
    # draw scene
    def draw(self, render = None, capture = False):

        # draw background
        frame_img = None
        progress = 0
        if render and capture:
            frame_img = render.copy()
            self.__surface.blit(frame_img, (0,0))
        elif render:
            progress = render.blit(self.__surface, (0, 0))
        else:
            self.__surface.fill(BLACK)

        # draw red line which indicates the progress of the rendering
        if render and render.in_progress(): 
            progress_len = int(self.__surface.get_width() * progress)
            pygame.draw.line(self.__surface, BLACK, (0, 0), (progress_len, 0), 1) 
            pygame.draw.line(self.__surface, RED, (0, 2), (progress_len, 2), 3) 
            pygame.draw.line(self.__surface, BLACK, (0, 4), (progress_len, 4), 1) 

        # update display
        pygame.display.flip()

        return frame_img

    #----------------------------------------------------------------------------------------------
    # init pygame diesplay surface
    def __init_surface(self, size):
        self.__surface = pygame.display.set_mode(size, pygame.RESIZABLE)

    #----------------------------------------------------------------------------------------------
    # handle events in a loop
    def __handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__run = False
            elif event.type == pygame.VIDEORESIZE:
                self.__init_surface((event.w, event.h))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.__run = False


###################################################################################################
# ray
class Ray:
    def __init__(self, a, b, ti = 0):
        self.A = a
        self.B = b
        self.__time = ti
    @property 
    def origin(self): 
        return self.A
    @origin.setter
    def origin(self, o): 
        self.A = o
    @property 
    def direction(self): 
        return self.B
    @direction.setter
    def direction(self, d): 
        self.B = d
    @property 
    def time(self): 
        return self.__time
    def point_at_parameter(self, t):
        return self.A + self.B * t


###################################################################################################
# axis aligned bounding box
class AABB:
    def __init__(self, a, b):
        self.__min = vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))
        self.__max = vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))
    @property 
    def min(self): 
        return self.__min
    @property 
    def max(self): 
        return self.__max
    def __or__(self, box1):
        box0 = self
        small = vec3(min(box0.min[0], box1.min[0]), min(box0.min[1], box1.min[1]), min(box0.min[2], box1.min[2]))
        big = vec3(max(box0.max[0], box1.max[0]), max(box0.max[1], box1.max[1]), max(box0.max[2], box1.max[2]))
        return AABB(small, big)
    def hit(self, r, tmin, tmax):
        for a in range(3):
            if r.direction[a] != 0:
                t0 = min((self.__min[a]-r.origin[a])/r.direction[a], (self.__max[a]-r.origin[a])/r.direction[a])
                t1 = max((self.__min[a]-r.origin[a])/r.direction[a], (self.__max[a]-r.origin[a])/r.direction[a])
                ttmin = max(t0, tmin)
                ttmax = min(t1, tmax)
                if ttmax < ttmin:
                    return False
        return True
        

###################################################################################################
# camera
class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist, t0=0, t1=0):
        self.__time0 = t0
        self.__time1 = t1
        self.__lens_radius = aperture/2
        self.__focus_dist = focus_dist
        self.__origin = lookfrom
        self.__direction = lookat -lookfrom
        self.__vup = vup
        self.__vfov = vfov * math.pi / 180
        self.__aspect = aspect
        self.update()
    @property
    def aspect(self):
        return self.__aspect
    @aspect.setter
    def aspect(self, aspect):
        self.__aspect = aspect
        self.update()
    @property
    def vfov_degree(self):
        return self.__vfov * 180/math.PI
    @vfov_degree.setter
    def vfov_degree(self, vfov):
        self.__vfov = vfov * math.pi/180
        self.update()
    def update(self):
        half_height = math.tan(self.__vfov/2)
        half_width = self.__aspect * half_height
        self.__w = -self.__direction.normalize()
        self.__u = self.__vup.cross(self.__w).normalize()
        self.__v = self.__w.cross(self.__u)
        self.__lower_left_corner = self.__origin - half_width*self.__focus_dist*self.__u - half_height*self.__focus_dist*self.__v - self.__focus_dist*self.__w
        self.__horizontal = 2*half_width*self.__focus_dist*self.__u
        self.__vertical = 2*half_height*self.__focus_dist*self.__v
    def get_ray(self, s, t):
        rd = self.__lens_radius*random_in_unit_disk()
        offset = self.__u*rd.x + self.__v*rd.y # dot(rd.xy, (u, v))
        time = self.__time0 + rand01() * (self.__time1-self.__time0)
        return Ray(
            self.__origin + offset,
            self.__lower_left_corner + s*self.__horizontal + t*self.__vertical - self.__origin - offset, time)


###################################################################################################
# hit information
class HitRecord:
    def __init__(self, t, p, normal, material):
        self.t = t
        self.p = p
        self.normal = normal
        self.material = material


###################################################################################################
# ray tracing object
class Hitable:
    def __init__(self):
        pass


###################################################################################################
# list of hitable objects
class HitableList(Hitable):
    def __init__(self):
        super().__init__()
        self.__list = []
    def __iadd__(self, hitobj):
        if type(hitobj)==list:
            self.__list.extend(hitobj)
        else:
            self.__list.append(hitobj)
        return self
    def append(self, hitobj):
        if type(hitobj)==list:
            self.__list.extend(hitobj)
        else:
            self.__list.append(hitobj)
    def bounding_box(self, t0, t1):
        if not self.__list:
            return None
        box = self.__list[0].bounding_box(t0, t1)
        if not box or len(self.__list) == 1:
            return box
        for i in range(1, len(self.__list)):
            box = box | self.__list[i].bounding_box(t0, t1)
            if not box:
                return box
        return box
    def hit(self, r, tmin, tmax):
        hit_anything, closest_so_far = None, tmax
        for hitobj in self.__list:
            rec = hitobj.hit(r, tmin, closest_so_far)
            if rec:
                hit_anything, closest_so_far = rec, rec.t
        return hit_anything
        

###################################################################################################
# Bonding Volume Hierarchy Node
class BHVNode:
    def __init__(self, hitobj, t0, t1):
        super().__init__()
        objlist = hitobj if type(hitobj)==list else [hitobj]
        if len(objlist) == 1:
            self.__left = self.__right = None
            self.__box = self.__left.bounding_box(self, t0, t1)
            return   
        axis = random.randrange(0, 3)          
        objlist.sort(key = lambda obj : obj.bounding_box(t0, t1).min[axis])
        half = len(objlist) - len(objlist) // 2
        self.__left = BHVNode(objlist[:half], t0, t1) if half != 1 else objlist[0]
        self.__right = BHVNode(objlist[half:], t0, t1) if half != len(objlist)-1 else objlist[-1]
        self.__box = self.__left.bounding_box(t0, t1) | self.__right.bounding_box(t0, t1)
    def bounding_box(self, t0, t1):
        return self.__box
    def hit(self, r, tmin, tmax):
        if not self.__box.hit(r, tmin, tmax):
            return None
        hit_left = self.__left.hit(r, tmin, tmax)
        hit_right = self.__right.hit(r, tmin, tmax)
        if hit_left and hit_right:
            return hit_left if hit_left.t < hit_right.t else hit_right 
        if hit_left:
            return hit_left
        if hit_right:
            return hit_right
        return None


###################################################################################################
# sphere hitable object
class Sphere(Hitable):
    def __init__(self, center, radius, material):
        super().__init__()
        self.__center = center
        self._radius = radius
        self.__material = material
    def center(self, time):
        return self.__center
    def bounding_box(self, t0, t1):
        return AABB(self.__center-vec3(self._radius, self._radius, self._radius), self.__center+vec3(self._radius, self._radius, self._radius))
    #----------------------------------------------------------------------------------------------
    # Ray - Sphere intersection
    #
    # Sphere:         dot(p-C, p-C) = R*R            `C`: center, `p`: point on the sphere, `R`, radius 
    # Ray:            p(t) = A + B * t               `A`: origin, `B`: direction        
    # Intersection:   dot(A +B*t-C, A+B*t-C) = R*R
    #                 t*t*dot(B,B) + 2*t*dot(B,A-C) + dot(A-C,A-C) - R*R = 0
    def hit(self, r, tmin, tmax):
        cpt = self.center(r.time)
        oc = r.origin - cpt
        a = r.direction.dot(r.direction)
        b = 2 * oc.dot(r.direction)
        c = oc.dot(oc) - self._radius*self._radius
        discriminant = b*b - 4*a*c
        if discriminant > 0:
            temp = (-b - math.sqrt(discriminant)) / (2*a)
            if tmin < temp < tmax:
                p = r.point_at_parameter(temp) 
                return HitRecord(temp, p, (p - cpt) / self._radius, self.__material)
            temp = (-b + math.sqrt(discriminant)) / (2*a)
            if tmin < temp < tmax:
                p = r.point_at_parameter(temp) 
                return HitRecord(temp, p, (p - cpt) / self._radius, self.__material)
        
        return None


###################################################################################################
# moving sphere hitable object
class MovingSphere(Sphere):
    def __init__(self, centers, times, radius, material):
        super().__init__(centers[0], radius, material)
        self.__centers = centers
        self.__times = times
    def center(self, time):
        if time < self.__times[0]:
            return self.__centers[0]
        for i, t in enumerate(self.__times):
            if t > time:
                delta_time = t-self.__times[i-1]
                if delta_time == 0:
                    return self.__centers[i]
                return self.__centers[i-1] + (self.__centers[i]-self.__centers[i-1])*(time-self.__times[i-1])/delta_time
        return self.__centers[-1]
    def bounding_box(self, t0, t1):
        small = vec3(min([c[0] for c in self.__centers]), min([c[1] for c in self.__centers]), min([c[2] for c in self.__centers]))
        small -= vec3(self._radius, self._radius, self._radius)
        big = vec3(max([c[0] for c in self.__centers]), max([c[1] for c in self.__centers]), max([c[2] for c in self.__centers]))
        big += vec3(self._radius, self._radius, self._radius)
        return AABB(small, big)


###################################################################################################
# noise 
class Noise:
    def __init__(self):
        pass


###################################################################################################
# noise 
class Perlin(Noise):
    def __init__(self):
        super().__init__()
        self.__rand = Perlin.generate()
        self.__perm = [Perlin.generate_perm() for _ in range(3)] 
    def noise(self, p):
        u, v, w = (p[i] - math.floor(p[i]) for i in range(3))
        i, j, k = (int(p[a]) for a in range(3))
        pe = self.__perm
        c = [[[self.__rand[pe[0][(i+di) & 255] ^ pe[1][(j+dj) & 255] ^ pe[2][(k+dk) & 255]] for dk in range(2)] for dj in range(2)] for di in range(2)]
        return Perlin.trilinear_interp(c, u, v, w)
    def turb(self, p, depth=7):
        accum = 0
        temp_p = vec3(p)
        weight = 1
        for i in range(3):
            accum += weight * self.noise(temp_p) 
            weight *= 0.5
            temp_p *= 2
        return math.fabs(accum)
    @staticmethod
    def generate():
        return [vec3(rand01(), rand01(), rand01() * 2 - 1).normalize() for _ in range(256)]
    @staticmethod
    def permute(p):
        for i in range(len(p)-1, 0, -1):
            target = int(rand01() * (i+1))
            p[i], p[target] = p[target], p[i] 
    @staticmethod
    def generate_perm():
        perm = [i for i in range(256)]
        Perlin.permute(perm)
        return perm
    @staticmethod
    def trilinear_interp(c, u, v, w):
        uu = u * u * (3 - 2 * u)
        vv = v * v * (3 - 2 * v)
        ww = w * w * (3 - 2 * w)
        accum = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    accum += \
                        ((i * uu) + (1 - i) * (1 - uu)) * \
                        ((j * vv) + (1 - j) * (1 - vv)) * \
                        ((k * ww) + (1 - k) * (1 - ww)) * \
                        c[i][j][k].dot(vec3(u-i, v-j, w-k))
        return accum

###################################################################################################
# texture
class Texture:
    def __init__(self):
        pass


###################################################################################################
# constant texture (color)
class ConstantTexture(Texture):
    def __init__(self, color):
        super().__init__()
        self.__color = color
    @staticmethod
    def create(r, g, b):
        return ConstantTexture(vec3(r, g, b))
    def value(self, u, v, p):
        return self.__color


###################################################################################################
# checker texture
class CheckerTexture(Texture):
    def __init__(self, even, odd):
        super().__init__()
        self.__even = even
        self.__odd = odd
    @staticmethod
    def create(r0, g0, b0, r1, g1, b1):
        return CheckerTexture(ConstantTexture.create(r0, g0, b0), ConstantTexture.create(r1, g1, b1))
    def value(self, u, v, p):
        sines = math.sin(10 * p.x) * math.sin(10 * p.y) * math.sin(10 * p.z)
        texture = self.__odd if sines < 0 else self.__even 
        return texture.value(0, 0, p)


###################################################################################################
# noise texture
class NoiseTexture(Texture):
    DEFAULT = 0
    TURB = 1
    SIN_X = 2
    SIN_Y = 3
    SIN_Z = 4 
    def __init__(self, scale, type):
        super().__init__()
        self.__noise = Perlin()
        self.__scale = scale
        self.__type = type
    @staticmethod
    def create(scale, type):
        return NoiseTexture(scale, type)
    def value(self, u, v, p):
        noise = 0.5
        if self.__type == NoiseTexture.DEFAULT:
            noise = self.__noise.noise(p * self.__scale)
        elif self.__type == NoiseTexture.TURB:
            noise = self.__noise.turb(p * self.__scale)
        elif self.__type == NoiseTexture.SIN_X:
            noise = math.sin(self.__scale * p.x + 5 * self.__noise.turb(p * self.__scale))
        elif self.__type == NoiseTexture.SIN_Y:
            noise = math.sin(self.__scale * p.y + 5 * self.__noise.turb(p * self.__scale))
        elif self.__type == NoiseTexture.SIN_Z:
            noise = math.sin(self.__scale * p.z + 10 * self.__noise.turb(p * self.__scale))
        else:
            noise = self.__noise.noise(p * self.__scale)
        return vec3(1, 1, 1) * (noise * 0.5 + 0.5)


###################################################################################################
# material
class Material:
    def __init__(self):
        pass


###################################################################################################
# lambertian material
class Lambertian(Material):
    def __init__(self, albedo):
        super().__init__()
        self.__albedo = albedo if albedo != None else ConstantTexture.create(0, 0, 0)
    def scatter(self, r_in, rec):
        # target is a point outside the sphere but "near" to `rec.p`: 
        # target = p + nv + random_direction
        target = rec.p + rec.normal + random_in_unit_sphere()
        return Ray(rec.p, target-rec.p, r_in.time), self.__albedo.value(0, 0, rec.p)
    

###################################################################################################
# metal material
class Metal(Material):
    def __init__(self, albedo, fuzz=0):
        super().__init__()
        self.__albedo = albedo if albedo != None else ConstantTexture.create(0, 0, 0)
        self.__fuzz = min(fuzz, 1)
    def scatter(self, r_in, rec):
        # reflection
        # reflected = reflect(r_in.direction.normalize(), rec.normal)
        reflected = r_in.direction.normalize().reflect(rec.normal)
        # fuzzy
        scattered = Ray(rec.p, reflected + self.__fuzz*random_in_unit_sphere(), r_in.time)
        attenuation = self.__albedo.value(0, 0, rec.p)
        return (scattered, attenuation) if scattered.direction.dot(rec.normal) > 0 else None


###################################################################################################
# dielectric material
class Dielectric(Material):
    def __init__(self, ri):
        super().__init__()
        self.__ref_idx = ri
    def scatter(self, r_in, rec):
        reflected = r_in.direction.reflect(rec.normal)
        if r_in.direction.dot(rec.normal) > 0:
            outward_normal = -rec.normal
            ni_over_nt = self.__ref_idx
            cosine = self.__ref_idx * r_in.direction.dot(rec.normal) / r_in.direction.magnitude()
        else:
            outward_normal = rec.normal
            ni_over_nt = 1/self.__ref_idx
            cosine = -r_in.direction.dot(rec.normal) / r_in.direction.magnitude()
        refracted = refract(r_in.direction, outward_normal, ni_over_nt)
        reflect_probe = schlick(cosine, self.__ref_idx) if refracted else 1
        if rand01() < reflect_probe:
            scattered = Ray(rec.p, reflected, r_in.time)
        else:
            scattered = Ray(rec.p, refracted, r_in.time)
        return scattered, vec3(1, 1, 1)


###################################################################################################
# render thread
class Rendering:
    def __init__(self, world, cam):
        self.__world = world
        self.__cam = cam
    #----------------------------------------------------------------------------------------------
    def start(self, size, no_sample, update_rate):
        self.__size = size
        self.__cam.aspect = self.__size[0]/self.__size[1]
        self.__no_samples = no_sample
        self.__update_rate = update_rate
        self.__pixel_count = 0
        self.__progress = 0
        self.__image = pygame.Surface(self.__size)
        self._stopped = False
        self.__thread = threading.Thread(target = self.run)
        self.__thread_lock = threading.Lock()
        self.__thread.start()
    #----------------------------------------------------------------------------------------------
    # check if thread is "running"
    def in_progress(self):
        return self.__thread.is_alive()
    #----------------------------------------------------------------------------------------------
    # wait for thread to end
    def wait(self, timeout = None):
        self.__thread.join(timeout)
    #----------------------------------------------------------------------------------------------
    # terminate
    def stop(self):
        self.__thread_lock.acquire() 
        self._stopped = True
        self.__thread_lock.release() 
        self.__thread.join()
    #----------------------------------------------------------------------------------------------
    # blit to surface
    def blit(self, surface, pos):
        self.__thread_lock.acquire()
        surface.blit(self.__image, pos) 
        progress = self.__progress
        self.__thread_lock.release() 
        return progress
    #---------------------------------------------------------------------------------------------- 
    # copy render surface
    def copy(self):
        self.__thread_lock.acquire() 
        image = self.__image.copy()
        self.__thread_lock.release() 
        return image 
    #----------------------------------------------------------------------------------------------
    def coord_iterator(self):
        max_s = max(*size)
        p2 = 0
        while max_s > 0:
            p2, max_s = p2 + 1, max_s >> 1
        tile_size = 1 << p2 -1
        yield (0, 0, *self.__size)
        self.__pixel_count = 1
        while tile_size > 0:
            no = (self.__size[0]-1) // tile_size + 1, (self.__size[1]-1) // tile_size + 1
            mx = (no[0]-1) // 2
            ix = [(mx - i//2) if i%2==0 else (mx+1+i//2) for i in range(no[0])]
            my = no[1] // 2
            iy = [(my + j//2) if j%2==0 else (my-1-j//2) for j in range(no[1])]
            for j in iy: 
                for i in ix:
                    if i % 2 != 0 or j % 2 != 0:
                       self.__pixel_count += 1
                    if tile_size >= 128 or i % 2 != 0 or j % 2 != 0:
                        x, y = i*tile_size, j*tile_size
                        w, h = min(tile_size, self.__size[0]-x), min(tile_size, self.__size[1]-y)
                        yield (x, y, w, h)
            tile_size >>= 1
    #----------------------------------------------------------------------------------------------
    def run(self):
        no_samples = self.__no_samples
        no_samples_outer = max(1, int(no_samples * self.__update_rate + 0.5))
        no_samples_inner = (no_samples + no_samples_outer - 1) // no_samples_outer
        outer_i = 0
        count = 0
        colarr = [0] * (self.__size[0] * self.__size[1] * 3)
        while outer_i * no_samples_inner < no_samples:
            iter = self.coord_iterator() 
            for x, y, w, h in iter:
                no_start = no_samples_inner * outer_i
                no_end = min(no_samples, no_start + no_samples_inner)
                col = vec3()
                for s in range(no_start, no_end):
                    u, v = (x + rand01())/self.__size[0], (y + rand01())/self.__size[1]
                    r = self.__cam.get_ray(u, v)
                    col += Rendering.rToColor(r, self.__world, 0)
                arri = y * self.__size[0] * 3 + x * 3
                colarr[arri+0] += col[0] 
                colarr[arri+1] += col[1]
                colarr[arri+2] += col[2] 
                col = vec3(colarr[arri+0], colarr[arri+1], colarr[arri+2])    

                self.__thread_lock.acquire()
                if no_start == 0:
                    surfaceSetXYWHf(self.__image, x, y, w, h, col / no_end)
                else:
                    surfaceSetXYWHf(self.__image, x, y, 1, 1, col / no_end)
                self.__thread_lock.release()
                count += 1
                self.__progress = count / (no_samples * self.__size[0] * self.__size[1])
                if self._stopped:
                    break
            outer_i += 1
    #----------------------------------------------------------------------------------------------
    max_dist = 1e20
    @staticmethod
    def rToColor(r, world, depth):
        rec = world.hit(r, 0.001, Rendering.max_dist) # 0.001 get rid of shadow acne
        if rec:
            if depth >= 50: # recursion depht
                return vec3(0, 0, 0) # TODO !!!
            sc_at = rec.material.scatter(r, rec)
            if not sc_at:
                return vec3(0, 0, 0)
            return multiply_components(sc_at[1], Rendering.rToColor(sc_at[0], world, depth+1))
            #return 0.5*vec3(rec.normal.x+1, rec.normal.y+1, rec.normal.z+1)
        unit_direction = r.direction.normalize()
        t = 0.5 * (unit_direction.y + 1)
        return (1-t)*vec3(1, 1, 1) + t*vec3(0.5, 0.7, 1)


###################################################################################################
# main

def random_scene(time0, time1):
    objlist = []
    objlist.append(Sphere(vec3(0, -1000, 0), 1000, Lambertian(CheckerTexture.create(0.2, 0.3, 0.1, 0.9, 0.9, 0.9))))

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = rand01()
            center = vec3(a+0.9*rand01(), 0.2, b+0.9*rand01())
            if (center-vec3(4, 0.2, 0)).magnitude() > 0.9:
                if choose_mat < 0.8:
                    # diffuse
                    mat = Lambertian(ConstantTexture.create(rand01()*rand01(), rand01()*rand01(), rand01()*rand01()))
                    if time0 != time1:
                        objlist.append(MovingSphere([center, center+vec3(0, 0.5*rand01(), 0)], [0, 1], 0.2, mat))  
                    else:
                        objlist.append(Sphere(center, 0.2, mat))  
                elif choose_mat < 0.95:
                    # metal
                    mat = Metal(ConstantTexture.create(0.5*(1+rand01()), 0.5*(1+rand01()), 0.5*(1+rand01())), 0.5*rand01())
                    objlist.append(Sphere(center, 0.2, mat))
                else:
                    # glass
                    mat = Dielectric(1.5)
                    objlist.append(Sphere(center, 0.2, mat))

    objlist.append(Sphere(vec3(0, 1, 0), 1, Dielectric(1.5)))
    objlist.append(Sphere(vec3(-4, 1, 0), 1, Lambertian(ConstantTexture.create(0.4, 0.2, 0.1))))
    objlist.append(Sphere(vec3(4, 1, 0), 1, Metal(ConstantTexture.create(0.7, 0.6, 0.5), 0.0)))
    world = BHVNode(objlist, time0, time1)
    return world

app = Application((600, 400), caption = "Ray Tracing: the Next Week")
    
size = app.size
image = pygame.Surface(size) 

scene_id = 4
time0, time1 = 0, 1
if scene_id == 0: # random scene

    lookfrom = vec3(12, 2, 3)
    lookat = vec3(0, 0, 0)
    dist_to_focus = 10
    #dist_to_focus = (lookat-lookfrom).magnitude()
    aperture = 0.1
    cam = Camera(lookfrom, lookat, vec3(0, 1, 0), 20, size[0]/size[1], aperture, dist_to_focus, time0, time1)
    world = random_scene(time0, time1)

elif scene_id == 1: # defocus blur

    lookfrom = vec3(3, 3, 2)
    lookat = vec3(0, 0, -1)
    dist_to_focus = (lookat-lookfrom).magnitude()
    aperture = 0.5
    cam = Camera(lookfrom, lookat, vec3(0, 1, 0), 20, size[0]/size[1], aperture, dist_to_focus)

    objlist = [
        Sphere(vec3(0, 0, -1), 0.5,      Lambertian(ConstantTexture.create(0.1, 0.2, 0.5))),
        Sphere(vec3(0, -100.5, -1), 100, Lambertian(ConstantTexture.create(0.8, 0.8, 0))),
        Sphere(vec3(1, 0, -1), 0.5,      Metal(ConstantTexture.create(0.8, 0.6, 0.2), 0.2)),
        Sphere(vec3(-1, 0, -1), 0.5,     Dielectric(1.5)),
        Sphere(vec3(-1, 0, -1), -0.45,   Dielectric(1.5))
    ] 
    world = BHVNode(objlist, 0, 0)

elif scene_id == 2: # motion blur

    lookfrom = vec3(4, 5, -4)
    lookat = vec3(0, 0, 0)
    dist_to_focus = (lookat-lookfrom).magnitude()
    aperture = 0.05
    cam = Camera(lookfrom, lookat, vec3(0, 1, 0), 20, size[0]/size[1], aperture, dist_to_focus, time0, time1)
    
    objlist = [
        Sphere(vec3(0, -100.5, 0), 100, Lambertian(ConstantTexture.create(0.8, 0.8, 0))),
        MovingSphere([vec3(-1.0, 0, 0.5), vec3(-0.5, 0, 0), vec3(-1.0, 0, -0.5)], [0, 0.5, 1], 0.5, Lambertian(ConstantTexture.create(0.1, 0.2, 0.5))),
        MovingSphere([vec3(0.5, 0, 0), vec3(0.5, 0, 0), vec3(1, 0, 0)], [0, 0.5, 1], 0.5, Metal(ConstantTexture.create(0.8, 0.6, 0.2), 0.2)),
        Sphere(vec3(0, 0, -1.0), 0.5,     Dielectric(1.5)),
        Sphere(vec3(0, 0, -1.0), -0.45,   Dielectric(1.5))
    ] 
    world = BHVNode(objlist, time0, time1)

elif  scene_id == 3: # textures

    lookfrom = vec3(13, 2, 3)
    lookat = vec3(0, 0, 0)
    dist_to_focus = 10
    aperture = 0
    cam = Camera(lookfrom, lookat, vec3(0, 1, 0), 20, size[0]/size[1], aperture, dist_to_focus, time0, time1)
    
    checker_texture = CheckerTexture.create(0.2, 0.3, 0.1, 0.9, 0.9, 0.9)
    objlist = [
        Sphere(vec3(0, -10, 0), 10, Lambertian(checker_texture)),
        Sphere(vec3(0, 10, 0), 10, Lambertian(checker_texture))
    ] 
    world = BHVNode(objlist, time0, time1)

else: # noise textures

    lookfrom = vec3(13, 2, 3)
    lookat = vec3(0, 0, 0)
    dist_to_focus = 10
    aperture = 0
    cam = Camera(lookfrom, lookat, vec3(0, 1, 0), 20, size[0]/size[1], aperture, dist_to_focus, time0, time1)
    
    checker_texture = NoiseTexture.create(1, NoiseTexture.SIN_Z)
    objlist = [
        Sphere(vec3(0, -1000, 0), 1000, Lambertian(checker_texture)),
        Sphere(vec3(0, 2, 0), 2, Lambertian(checker_texture))
    ] 
    world = BHVNode(objlist, time0, time1)

render = Rendering(world, cam)
app.run(render, 100, 1, 0)
    
    
    

    