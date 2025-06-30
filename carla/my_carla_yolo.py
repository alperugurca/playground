import time
import carla
import pygame
import numpy as np
import heapq
import math
import cv2
from ultralytics import YOLO

vehicle = None
camera = None
yolo_camera = None
display = None
clock = None
path= None
path_index = 0
debug= None

current_frame = None
yolo_detections = []
vision_obstacles = []

CROP_TOP_RATIO = 0.4     # Üstten %40'unu kes (gökyüzü ve uzak nesneler)
CROP_BOTTOM_RATIO = 0.9  # Alttan %90'ına kadar al
CROP_LEFT_RATIO = 0.4    # Soldan %40'unu kes
CROP_RIGHT_RATIO = 0.6   # Sağdan %60'ına kadar al
IM_WIDTH = 640
IM_HEIGHT = 480

START_DELAY_SECONDS = 5 # Başlangıç gecikmesi
start_time =0.0
movement_started = False

DANGEEROUS_OBJECTS = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'person']

class Node:
    def __init__(self, waypoint, f_score=float('inf')):
        self.waypoint = waypoint
        self.f_score = f_score
        
    def __lt__(self, other):
        return self.f_score < other.f_score

def heuristic(a, b):
    return a.distance(b)

def astar(start, goal):
    open_set = [Node(start, 0)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start.transform.location, goal)}

    while open_set:
        current_node = heapq.heappop(open_set)
        current = current_node.waypoint

        if current.transform.location.distance(goal) < 1.0:
            path = []
            while current in came_from:
                path.append(current.transform.location)
                current = came_from[current]

            path.append(start.transform.location)
            return path[::-1]
        
        for next_wp in current.next(2.0):
            if next_wp.lane_type != carla.LaneType.Driving:
                continue
            neighbor = next_wp
            tentative_g_score = g_score[current] + current.transform.location.distance(neighbor.transform.location)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor.transform.location, goal)
                heapq.heappush(open_set, Node(neighbor, f_score[neighbor]))
    return None

def draw_waypoints(world, path, life_time=0.5):
    """Yolu görselleştir"""
    global debug
    if debug is None:
        debug = world.debug
    
    if path:
        for i in range(len(path) - 1):
            debug.draw_line(
                path[i],
                path[i + 1],
                thickness=0.2,
                color=carla.Color(0, 255, 0),
                life_time=life_time
            )

def process_yolo_image(image):
    global current_frame
    i=np.array(image.raw_data)
    i2=i.reshape(IM_HEIGHT, IM_WIDTH, 4)    
    i3=i2[:,:,:3]

    crop_start_y= int(IM_HEIGHT * CROP_TOP_RATIO)
    crop_end_y= int(IM_HEIGHT * CROP_BOTTOM_RATIO)
    crop_start_x= int(IM_WIDTH * CROP_LEFT_RATIO)
    crop_end_x= int(IM_WIDTH * CROP_RIGHT_RATIO)

    cropped=i3[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
    current_frame=cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    return cropped/255.0

def analyze_yolo_detections(model):
    global yolo_detections, vision_obstacles, current_frame

    if current_frame is None:
        return False, []
    
    try:
        results=model(current_frame, stream=False, verbose=False)

        vision_obstacles=[]
        emergency_detected=False
        detected_object=[]

        for result in results:
            if result.boxes is not None:
                boxes=result.boxes
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    conf=float(boxes.conf[i]) if boxes.conf is not None else 1.0
                    cls=int(boxes.cls[i]) if boxes.cls is not None else  0
                    class_name=model.names[cls] if cls < len(model.names) else f"Class {cls}"

                    if conf < 0.5:
                        continue

                    detected_object.append({
                        'name': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'center_x': (x1 + x2) // 2,
                        'center_y': (y1 + y2) // 2,
                        'width': x2 - x1,
                        'height': y2 - y1
                    })

                    if class_name in DANGEEROUS_OBJECTS:

                        frame_height, frame_width = current_frame.shape[:2]
                        center_x=(x1+x2) // 2
                        center_y=(y1+y2) // 2

                        distance_factor=(frame_height - center_y) / frame_height

                        size_factor=((x2 -x1)* (y2-y1))/ (frame_width * frame_height)
                        danger_score=(size_factor * 0.6 + distance_factor * 0.4) * conf

                        vision_obstacles.append({
                            'object': class_name,
                            'confidence': conf,
                            'danger_score': danger_score,
                            'position': (center_x, center_y),
                            'bbox': (x1, y1, x2, y2)
                        })

                        if danger_score > 0.6 or (class_name in ['person, biycle', 'motorcycle'] and danger_score > 0.4):
                            emergency_detected=True
        yolo_detections = detected_object
        return emergency_detected, vision_obstacles
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return False, []

def draw_yolo_detections():
    global current_frame, yolo_detections
    if current_frame is None or not yolo_detections:
        return current_frame
    
    annotated_frame=current_frame.copy()

    for detection in yolo_detections:
        x1, y1, x2, y2 = detection['bbox']
        conf=detection['confidence']
        name=detection['name']

        color = (0,0,255) if name in DANGEEROUS_OBJECTS else (0, 255, 0)

        cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), color, 2)

        label= f'{name}: {conf:.2f}'
        cv2.putText(annotated_frame, label, (x1, y1 -10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_frame

def calculate_vision_based_speed(base_speed):
    global vision_obstacles

    if not vision_obstacles:
        return base_speed
    
    max_danger= max(obs['danger_score'] for obs in vision_obstacles)

    if max_danger > 0.6:
        return 0.0
    
    elif max_danger > 0.4:
        return base_speed * 0.2
    
    elif max_danger > 0.1:
        return base_speed * 0.5
    
    else:
        return base_speed

def carla_image_to_pygame_surface(carla_image):
    array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (carla_image.height, carla_image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return pygame.surfarray.make_surface(array.swapaxes(0, 1))


def calculate_steering_angle(current_location, target_location):
    global vehicle
    if target_location is None or current_location is None or vehicle is None:
        return 0.0

    target_vector = target_location - current_location
    current_forward_vector = vehicle.get_transform().get_forward_vector()

    target_vector_2d = np.array([target_vector.x, target_vector.y])
    current_vector_2d = np.array([current_forward_vector.x, current_forward_vector.y])
    
    if np.linalg.norm(target_vector_2d) > 0:
        target_vector_2d = target_vector_2d / np.linalg.norm(target_vector_2d)
    if np.linalg.norm(current_vector_2d) > 0:
        current_vector_2d = current_vector_2d / np.linalg.norm(current_vector_2d)
    
    cross_product = np.cross(current_vector_2d, target_vector_2d)
    dot_product = np.dot(current_vector_2d, target_vector_2d)
    
    angle = np.degrees(np.arctan2(cross_product, dot_product))
    return angle

def get_speed(vehicle):
    vel= vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


def show_countdown(seconds_left):
    global display
    if display:
        display.fill((0, 0, 0))

        font=pygame.font.SysFont('Arial', 72, bold=True)
        small_font=pygame.font.SysFont('Arial', 36)

        if seconds_left > 0:
            countdown_text=font.render(str(int(seconds_left)),True, (255, 255, 255))
            status_text=small_font.render("Hareket için bekleniyor...", True, (255, 255, 0))
        
        else:
            countdown_text=font.render("BAŞLIYOR!", True, (0, 255, 0))
            status_text=small_font.render("Otonom sürüş başlıyor...", True, (0, 255, 0))
        
        countdown_rect=countdown_text.get_rect(center=(IM_WIDTH // 2, IM_HEIGHT // 2))
        status_rect=status_text.get_rect(center=(IM_WIDTH // 2, IM_HEIGHT // 2 + 50))

        display.blit(countdown_text, countdown_rect)
        display.blit(status_text, status_rect)
        pygame.display.flip()

def simple_throttle_control(target_speed, current_speed):
    speed_diff=target_speed - current_speed

    if speed_diff > 0:
        if speed_diff > 5.0:
            return 0.8
        elif speed_diff > 2.0:
            return 0.5
        else:
            return 0.3
    else:
        return 0.0  

def simple_brake_control (target_speed, current_speed):

    speed_diff=current_speed - target_speed

    if speed_diff > 0:
        if speed_diff > 5.0:
            return 0.8
        elif speed_diff > 2.0:
            return 0.5
        else:
            return 0.3
    else:
        return 0.0 

def simple_steering_control(target_angle):
    if target_angle > 180:
        target_angle -= 360
    elif target_angle < -180:
        target_angle += 360

    steer_factor=target_angle / 90.0
    return np.clip(steer_factor * 0.5, -1.0, 1.0)

def main():
    global vehicle, camera, yolo_camera, display, clock, path, path_index, start_time, movement_started

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    map=world.get_map()
    debug=world.debug

    try:
        model=YOLO('yolov8n.pt')
    
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = map.get_spawn_points()

        start_waypoint = map.get_waypoint(spawn_points[1].location)
        end_waypoint = map.get_waypoint(spawn_points[3].location)

        vehicle = world.spawn_actor(vehicle_bp, spawn_points[1])

        path = astar(start_waypoint, end_waypoint.transform.location) 

        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=-5, y=0, z=2.5), carla.Rotation(pitch=-15))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        yolo_camera_bp = bp_lib.find('sensor.camera.rgb')
        yolo_camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        yolo_camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        yolo_camera_bp.set_attribute('fov', '90')  # Geniş görüş açısı
        yolo_camera_transform = carla.Transform(carla.Location(x=2.5, y=0, z=1.2))
        yolo_camera = world.spawn_actor(yolo_camera_bp, yolo_camera_transform, attach_to=vehicle)

        pygame.init()
        display = pygame.display.set_mode((800, 600))
        clock=pygame.time.Clock()

        def process_main_camera(image):
            global movement_started, start_time

            if not movement_started:
                return
            
            surface=carla_image_to_pygame_surface(image)
            display.blit(surface,(0, 0))

            font=pygame.font.SysFont('Arial', 18)
            current_speed=get_speed(vehicle)
            speed_text=font.render(f"Hız: {current_speed*3.6:.1f} km/s", True, (255, 255, 255))
            display.blit(speed_text, (10, 10))

            y_offset=40
            if yolo_detections:
                detection_text=font.render(f"Algılanan nesneler: {len(yolo_detections)}", True, (255, 255, 0))
                display.blit(detection_text, (10, y_offset))
                y_offset += 25

                for i, detection in enumerate(yolo_detections[:3]):
                    obj_text=font.render(f"{detection['name']}: {detection['confidence']:.2f}", True, (255, 255, 255))
                    display.blit(obj_text, (10,y_offset))
                    y_offset += 20

            if vision_obstacles:
                max_danger=max(obs['danger_score'] for obs in vision_obstacles)
                if max_danger > 0.6:
                    danger_text=font.render("Acil durum! Yüksek tehlike!", True, (255, 0, 0))
                elif max_danger > 0.4:
                    danger_text=font.render("Dikkat! Orta tehlike!", True, (255, 165, 0))
                else:
                    danger_text=font.render("Güvenli sürüş", True, (255, 255, 0))
                display.blit(danger_text, (10, y_offset))
                y_offset += 25
            
            if vision_obstacles:
                highest_risk=max(vision_obstacles, key=lambda x: x['danger_score'])
                risk_text=font.render(f"En yüksek risk: {highest_risk['object']} ({highest_risk['danger_score']:.2f})", True, (255, 255, 255))
                display.blit(risk_text, (10, y_offset))
            
            pygame.display.flip()
        
        camera.listen(process_main_camera)
        yolo_camera.listen(lambda data: process_yolo_image(data))

        draw_waypoints(world, path)

        max_speed=8.0 # Maksimum hız 8 m/s (28.8 km/s)
        last_time=time.time()

        start_time=time.time()
        movement_started=False

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key ==pygame.K_ESCAPE:
                        movement_started=True
            
            now=time.time()
            dt=now - last_time

            if dt <=0:
                continue
            
            if not movement_started:
                elapsed_time=now - start_time
                remaining_time=START_DELAY_SECONDS - elapsed_time

                if remaining_time <= 0:
                    movement_started=True
                    print("Hareket başlıyor!")
                else:
                    show_countdown(remaining_time)
                    
                    analyze_yolo_detections(model)

                    if current_frame is not None:
                        annotated_frame=draw_yolo_detections()
                        cv2.imshow("YOLO Detections", annotated_frame)
                        key=cv2.waitKey(1) & 0xFF
                        if key == 27:
                            moevement_started=True
                    
                    clock.tick(30)
                    continue
            current_location=vehicle.get_location()
            current_speed=get_speed(vehicle)

            vision_emergency,_=analyze_yolo_detections(model)

            if current_frame is not None:
                annotated_frame=draw_yolo_detections()
                cv2.imshow("YOLO Detections", annotated_frame)
                key=cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            
            target_speed=max_speed
            emergency_brake=False

            vision_safe_speed=calculate_vision_based_speed(max_speed)
            target_speed=vision_safe_speed

            if vision_emergency:
                emergency_brake=True
                target_speed=0.0
                print("Acil durum algılandı! Hız sıfırlanıyor.")
            
            if path and path_index < len(path):
                target_location=path[path_index]
                distance_to_target=current_location.distance(target_location)

                if distance_to_target < max(2.0, current_speed * 0.5):
                    path_index += 1
                    if path_index < len(path):
                        debug.draw_point(path[path_index], size=0.1, color=carla.Color(255,0,0), life_time=0.5)    
                    continue

                target_angle=calculate_steering_angle(current_location, target_location)
                steer=simple_steering_control(target_angle)

                throttle = 0.0
                brake=0.0
                
                if emergency_brake:
                    throttle =0.0
                    brake = 1.0
                else:
                    if current_speed > target_speed:
                        brake=simple_brake_control(target_speed, current_speed)
                        throttle = 0.0
                    else:
                        brake=0.0
                        throttle=simple_throttle_control(target_speed, current_speed)
                
                control=carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
                vehicle.apply_control(control)

                if debug:
                    debug.draw_point(target_location, size=0.1, color=carla.Color(0,0,255), life_time=0.1)
            else:
                print("Yol tamamlandı.")
                break

            clock.tick(30)
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        cv2.destroyAllWindows()
        if camera:
            camera.destroy()
        if yolo_camera:
            yolo_camera.destroy()
        if vehicle:
            vehicle.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()


                

