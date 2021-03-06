# Raytrace renderer
# 
# Implemented for PyGame [https://www.pygame.org]
#
# Based on:
#  
# "Ray Tracing in One Weekend (Ray Tracing Minibooks Book 1)" by Peter Shirley
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
                print("Render time:", (current_time-start_time)/1000, " seconds" )

        self.__render_image = render.copy()
        writeFilePPM(self.__render_image, "rt_1")
        pygame.image.save(self.__render_image, "rt_1.png")

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
    def __init__(self, a, b):
        self.A = a
        self.B = b
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
    def point_at_parameter(self, t):
        return self.A + self.B * t


###################################################################################################
# camera
class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist):
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
        return Ray(
            self.__origin + offset,
            self.__lower_left_corner + s*self.__horizontal + t*self.__vertical - self.__origin - offset)


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
    def hit(self, r, tmin, tmax):
        hit_anything, closest_so_far = None, tmax
        for hitobj in self.__list:
            rec = hitobj.hit(r, tmin, closest_so_far)
            if rec:
                hit_anything, closest_so_far = rec, rec.t
        return hit_anything


###################################################################################################
# sphere hitable object
class Sphere(Hitable):
    def __init__(self, center, radius, material):
        super().__init__()
        self.__center = center
        self.__radius = radius
        self.__material = material
    #----------------------------------------------------------------------------------------------
    # Ray - Sphere intersection
    #
    # Sphere:         dot(p-C, p-C) = R*R            `C`: center, `p`: point on the sphere, `R`, radius 
    # Ray:            p(t) = A + B * t               `A`: origin, `B`: direction        
    # Intersection:   dot(A +B*t-C, A+B*t-C) = R*R
    #                 t*t*dot(B,B) + 2*t*dot(B,A-C) + dot(A-C,A-C) - R*R = 0
    def hit(self, r, tmin, tmax):
        oc = r.origin - self.__center
        a = r.direction.dot(r.direction)
        b = 2 * oc.dot(r.direction)
        c = oc.dot(oc) - self.__radius*self.__radius
        discriminant = b*b - 4*a*c
        if discriminant > 0:
            temp = (-b - math.sqrt(discriminant)) / (2*a)
            if tmin < temp < tmax:
                p = r.point_at_parameter(temp) 
                return HitRecord(temp, p, (p - self.__center) / self.__radius, self.__material )
            temp = (-b + math.sqrt(discriminant)) / (2*a)
            if tmin < temp < tmax:
                p = r.point_at_parameter(temp) 
                return HitRecord(temp, p, (p - self.__center) / self.__radius, self.__material )
        
        return None


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
        self.__albedo = albedo
    def scatter(self, r_in, rec):
        # target is a point outside the sphere but "near" to `rec.p`: 
        # target = p + nv + random_direction
        target = rec.p + rec.normal + random_in_unit_sphere()
        return Ray(rec.p, target-rec.p), self.__albedo
    

###################################################################################################
# metal material
class Metal(Material):
    def __init__(self, albedo, fuzz=0):
        super().__init__()
        self.__albedo = albedo
        self.__fuzz = min(fuzz, 1)
    def scatter(self, r_in, rec):
        # reflection
        # reflected = reflect(r_in.direction.normalize(), rec.normal)
        reflected = r_in.direction.normalize().reflect(rec.normal)
        # fuzzy
        scattered = Ray(rec.p, reflected + self.__fuzz*random_in_unit_sphere())
        attenuation = self.__albedo
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
            scattered = Ray(rec.p, reflected)
        else:
            scattered = Ray(rec.p, refracted)
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

def random_scene():
    list = HitableList()
    list.append(Sphere(vec3(0, -1000, 0), 1000, Lambertian(vec3(0.5, 0.5, 0.5))))

    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = rand01()
            center = vec3(a+0.9*rand01(), 0.2, b+0.9*rand01())
            if (center-vec3(4, 0.2, 0)).magnitude() > 0.9:
                if choose_mat < 0.8:
                    # diffuse
                    mat = Lambertian(vec3(rand01()*rand01(), rand01()*rand01(), rand01()*rand01()))
                    list.append(Sphere(center, 0.2, mat))  
                elif choose_mat < 0.95:
                    # metal
                    mat = Metal(vec3(0.5*(1+rand01()), 0.5*(1+rand01()), 0.5*(1+rand01())), 0.5*rand01())
                    list.append(Sphere(center, 0.2, mat))
                else:
                    # glass
                    mat = Dielectric(1.5)
                    list.append(Sphere(center, 0.2, mat))

    list.append(Sphere(vec3(0, 1, 0), 1, Dielectric(1.5)))
    list.append(Sphere(vec3(-4, 1, 0), 1, Lambertian(vec3(0.4, 0.2, 0.1))))
    list.append(Sphere(vec3(4, 1, 0), 1, Metal(vec3(0.7, 0.6, 0.5), 0.0)))
    return list


app = Application((600, 400), caption = "Ray Tracing in One Weekend")
    
size = app.size
image = pygame.Surface(size) 

create_random_scene = True
if create_random_scene:

    lookfrom = vec3(12, 2, 3)
    lookat = vec3(0, 0, 0)
    dist_to_focus = 10
    #dist_to_focus = (lookat-lookfrom).magnitude()
    aperture = 0.1
    cam = Camera(lookfrom, lookat, vec3(0, 1, 0), 20, size[0]/size[1], aperture, dist_to_focus)
    world = random_scene()

else:

    lookfrom = vec3(3, 3, 2)
    lookat = vec3(0, 0, -1)
    dist_to_focus = (lookat-lookfrom).magnitude()
    aperture = 0.5
    cam = Camera(lookfrom, lookat, vec3(0, 1, 0), 20, size[0]/size[1], aperture, dist_to_focus)

    world = HitableList()
    world += [
        Sphere(vec3(0, 0, -1), 0.5,      Lambertian(vec3(0.1, 0.2, 0.5))),
        Sphere(vec3(0, -100.5, -1), 100, Lambertian(vec3(0.8, 0.8, 0))),
        Sphere(vec3(1, 0, -1), 0.5,      Metal(vec3(0.8, 0.6, 0.2), 0.2)),
        Sphere(vec3(-1, 0, -1), 0.5,     Dielectric(1.5)),
        Sphere(vec3(-1, 0, -1), -0.45,   Dielectric(1.5))
    ] 

render = Rendering(world, cam)
app.run(render, 100, 1, 0)
    
    
    

    