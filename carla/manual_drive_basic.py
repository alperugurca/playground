import random
import time
import carla
import pygame
from pygame.locals import *

reverse=False
gear=1
last_reverse_toggle_time=0

def process_input(keys, vehicle):
    global reverse, gear, last_reverse_toggle_time
    if vehicle is None:
        return
    throttle = 0.0
    steer=0.0

    if keys[K_UP]:
        throttle=1.0
    elif keys[K_DOWN]:
        throttle=throttle-throttle*0.1
    elif keys[K_LEFT]:
        steer=-1.0
    elif keys[K_RIGHT]:
        steer=1.0

    if keys[K_r]:
        current_time=time.time()
        if current_time - last_reverse_toggle_time > 0.5:
            gear=gear*-1
            if gear==1:
                reverse=False
            else:
                reverse=True
            last_reverse_toggle_time=current_time
    print('Current Gear:', 'Reverse' if reverse else 'Forward')
    control=carla.VehicleControl(throttle=throttle, steer=steer, reverse=reverse)
    vehicle.apply_control(control)

def main():
    client=carla.Client('localhost', 2000)
    world=client.get_world()

    try:
        bp_lib=world.get_blueprint_library()

        vehicle_bp=bp_lib.find('vehicle.audi.tt')
        spawn_point=random.choice(world.get_map().get_spawn_points())
        vehicle=world.spawn_actor(vehicle_bp, spawn_point)

        pygame.init()
        pygame.display.set_mode((640, 480))
        clock=pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type==QUIT:
                    return
            keys=pygame.key.get_pressed()
            process_input(keys, vehicle)
            world.tick()
            clock.tick(60)



    finally:
        if vehicle is not None:
            vehicle.destroy()
            pygame.quit()

if __name__=='__main__':
    main()