import random
import time
import carla
import pygame
from pygame.locals import *
# from carla import ColorConverter as cc
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import threading
from matplotlib import cm
import open3d as o3d

reverse=False
gear=1
last_reverse_toggle_time=0

IM_WIDTH=640
IM_HEIGHT=480
# actor_list=[]



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
    # print('Current Gear:', 'Reverse' if reverse else 'Forward')
    control=carla.VehicleControl(throttle=throttle, steer=steer, reverse=reverse)
    vehicle.apply_control(control)

# def process_image(image):
#     image.convert(cc.CityScapesPalette) # change to cityscapes palette
#     i=np.array(image.raw_data)
#     i2=i.reshape((IM_HEIGHT, IM_WIDTH, 4))
#     i3=i2[:, :, :3]
#     cv2.imshow('', i3)
#     cv2.waitKey(1)
#     return i3/255.0

# def carla_image_to_pygame_surface(carla_image):
#     array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
#     array = np.reshape(array, (carla_image.height, carla_image.width, 4))
#     array = array[:, :, :3]
#     array = array[:, :, ::-1]
#     return pygame.image.frombuffer(array.copy(), (carla_image.width, carla_image.height), 'RGB')

# def process_gnss_data(gnss_event):
#     global latitude, longitude
#     latitude.append(gnss_event.latitude)
#     longitude.append(gnss_event.longitude)
#     print(f'Latitude: {gnss_event.latitude}, Longitude: {gnss_event.longitude}')

# def update_plot(frame):
#     plt.cla()
#     plt.plot(longitude, latitude, 'b-', marker='o')
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title('GNSS Data')
#     plt.grid(True)

# def run_matplotlib():
#     fig,_ = plt.subplots()
#     ani = animation.FuncAnimation(fig, update_plot, interval=100)
#     plt.show()

VIRIDIS=np.array(cm.get_cmap('plasma').colors)
VID_RANGE=np.linspace(0.0, 1.0, VIRIDIS.shape[0])








def main():

    # global latitude, longitude
    # latitude=[]
    # longitude=[]

    # matplotlib_thread=threading.Thread(target=run_matplotlib)
    # matplotlib_thread.start()

    client=carla.Client('localhost', 2000)
    world=client.get_world()

    try:
        bp_lib=world.get_blueprint_library()

        vehicle_bp=bp_lib.find('vehicle.audi.tt')
        spawn_point=random.choice(world.get_map().get_spawn_points())
        vehicle=world.spawn_actor(vehicle_bp, spawn_point)

        pygame.init()
        display=pygame.display.set_mode((800, 600))
        clock=pygame.time.Clock()

        camera_bp=bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '110')


        lidar_bp=bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('rotation_frequency', '100')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '500000')


        def process_lidar_data(lidar_data):
            points=np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
            points=points.reshape((-1, 4))
            points=points[:, :3]


        # imu_bp=bp_lib.find('sensor.other.imu')
        # imu_transform=carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0), carla.Rotation(pitch=0, yaw=0, roll=0))
        # imu_sensor=world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)



        # gnss_bp=bp_lib.find('sensor.other.gnss')
        # gnss_sensor=world.spawn_actor(gnss_bp, carla.Transform(carla.Location(z=2.5)), attach_to=vehicle)

        # front_camera=bp_lib.find('sensor.camera.semantic_segmentation')
        # front_camera.set_attribute('image_size_x', str(IM_WIDTH))
        # front_camera.set_attribute('image_size_y', str(IM_HEIGHT))
        # front_camera.set_attribute('fov', '110')
        # front_camera_spawn_point=carla.Transform(carla.Location(x=2.5, z=0.7))
        # front_camera_sensor=world.spawn_actor(front_camera, front_camera_spawn_point, attach_to=vehicle)
        # actor_list.append(front_camera_sensor)
        

        # collision=bp_lib.find('sensor.other.collision')
        # lane_invasion=bp_lib.find('sensor.other.lane_invasion')
        # obstacle=bp_lib.find('sensor.other.obstacle')

        # obstacle.set_attribute('distance', '3')

        # collision_sensor=world.spawn_actor(collision, carla.Transform(), attach_to=vehicle)
        # lane_invasion_sensor=world.spawn_actor(lane_invasion, carla.Transform(), attach_to=vehicle)
        # obstacle_sensor=world.spawn_actor(obstacle, carla.Transform(carla.Location(x=2.5, z=1.0)), attach_to=vehicle)

        # actor_list.append(collision_sensor)
        # actor_list.append(lane_invasion_sensor)
        # actor_list.append(obstacle_sensor)

        # collision_sensor.listen(lambda event: print('Collision Detected', event))

        # def on_invasion(event):
        #     print('Lane Invasion Detected', event)
        #     for mark in event.crossed_lane_markings:
        #         print('Lane invasion detected', mark.type)
        # lane_invasion_sensor.listen(on_invasion)

        # def on_obstacle_detected(event):
        #     print('Obstacle Detected!')
        #     print(f'Distance: {event.distance:.2f} meters')

        #     if event.other_actor:
        #         print(f'Actor: {event.other_actor.type_id}')
        #     else:
        #         print('No actor info available')

        # obstacle_sensor.listen(on_obstacle_detected)

        # def imu_callback(data):
        #     print('IMU Data:')
        #     print('Acceleration:', data.accelerometer)
        #     print('Gyroscope:', data.gyroscope)
        #     print('Magnetometer:', data.compass)
        #     time.sleep(5)

        # imu_sensor.listen(imu_callback)
        camera_transform=carla.Transform(carla.Location(x=-5, z=2.0), carla.Rotation(pitch=-15))
        camera_sensor=world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        lidar_transform=carla.Transform(carla.Location(x=0, z=2.5))
        lidar_sensor=world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        point_list=o3d.geometry.PointCloud()

        def add_open3d_axis(vis):
            axis=o3d.geometry.LineSet()
            axis.points=o3d.utility.Vector3dVector(np.array([
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]))
            axis.lines=o3d.utility.Vector2iVector(np.array([
                [0, 1],
                [0, 2],
                [0, 3]
            ]))
            axis.colors=o3d.utility.Vector3dVector(np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]))
            vis.add_geometry(axis)
            return vis
        
        def lidar_callback(point_cloud,point_list):
            data=np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
            data=data.reshape(data,(int(data.shape[0]/4), 4))

            intensity=data[:, -1]
            intensity_col=1.0-np.log(intensity)/np.log(np.exp(-0.004*100))
            int_color=np.c_[
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])
            ]

            points=data[:, :-1]
            points[:, 1]=-points[:, 1]

            points_list=o3d.utility.Vector3dVector(points)
            colors_list=o3d.utility.Vector3dVector(int_color)



        # def process_camera_data(image):
        #     surface=carla_image_to_pygame_surface(image)
        #     display.blit(surface, (0, 0))
        #     pygame.display.flip()

        lidar_sensor.listen(lambda data: lidar_callback(data, point_list))
        # camera_sensor.listen(process_camera_data)

        vis=o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Lidar Point Cloud',
            width=960,
            height=540,
            left=480,
            top=270,
            visible=True
        )

        vis.get_render_option().background_color=[0.05, 0.05, 0.05]
        vis.get_render_option().point_size=2.0
        vis.get_render_option().show_coordinate_frame=True
        add_open3d_axis(vis)

        frame=0
        while True:
            if frame==2:
                vis.update_geometry(point_list)

            frame+=1

            vis.update_geometry(point_list)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.005)


        # gnss_sensor.listen(process_gnss_data)
        # front_camera_sensor.listen(lambda data: process_image(data))


        # while True:
        #     for event in pygame.event.get():
        #         if event.type==QUIT:
        #             return
        #     keys=pygame.key.get_pressed()
        #     process_input(keys, vehicle)
        #     world.tick()
        #     clock.tick(60)



    finally:
        if vehicle is not None:
            vehicle.destroy()
            pygame.quit()

if __name__=='__main__':
    main()