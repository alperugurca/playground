import time
import carla
import pygame
import numpy as np  
import heapq
import math

# Global değişkenler
vehicle = None
camera = None
display = None
clock = None
path = None
path_index = 0
obstacle_distance = None
obstacle_actor = None
debug = None

class PIDController:
    def __init__(self, Kp=0.5, Ki=0.05, Kd=0.1):
        self.Kp = Kp  # Orantısal katsayı
        self.Ki = Ki  # İntegral katsayı
        self.Kd = Kd  # Türev katsayı
        self.integral = 0.0
        self.prev_error = 0.0
        self.min_output = 0.0
        self.max_output = 1.0

    def get_pid_params(self, error):
        abs_error = abs(error)
        if abs_error < 5:
            return 0.1, 0.01, 0.02
        elif abs_error < 15:
            return 0.3, 0.02, 0.05
        else:
            return 0.6, 0.03, 0.1

    def run_step(self, target_speed, current_speed, dt):
        error = target_speed - current_speed
        
        # İntegral birikimi sınırlama (anti-windup)
        if self.integral * error < 0:
            # Eğer hata değişti ise integral sıfırla
            self.integral = 0
        
        self.integral += error * dt
        # İntegral değerini makul bir aralıkta tut
        self.integral = np.clip(self.integral, -10.0, 10.0)
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        Kp, Ki, Kd = self.get_pid_params(error)
        output = Kp * error + Ki * self.integral + Kd * derivative
        
        return np.clip(output, self.min_output, self.max_output)


class ThrottleController(PIDController):
    def __init__(self):
        super().__init__(Kp=0.5, Ki=0.05, Kd=0.1)
        self.min_output = 0.0
        self.max_output = 1.0


class BrakeController(PIDController):
    def __init__(self):
        super().__init__(Kp=0.7, Ki=0.05, Kd=0.1)
        self.min_output = 0.0
        self.max_output = 1.0
    
    def run_step(self, target_speed, current_speed, dt):
        # Fren için hata: mevcut hız - hedef hız (durmak için)
        error = current_speed - target_speed
        return super().run_step(error, 0, dt) if error > 0 else 0.0

class SteeringController:
    def __init__(self):
        self.prev_error = 0.0
        self.Kp = 0.8
        self.Kd = 0.2
        
    def run_step(self, current_angle, target_angle, dt):
        error = target_angle - current_angle
        # Açıyı -180 ile 180 derece arasında normalize et
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360
            
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        
        output = self.Kp * (error / 90.0) + self.Kd * derivative
        return np.clip(output, -1.0, 1.0)

class Node:
    def __init__(self, waypoint, f_score=float('inf')):
        self.waypoint = waypoint
        self.f_score = f_score
        
    def __lt__(self, other):
        return self.f_score < other.f_score

def is_obstacle_in_front(vehicle, obstacle_location):
    """Engelin aracın önünde olup olmadığını kontrol et."""
    if not vehicle:
        return False
    
    # Aracın dönüş bilgilerini al
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_forward = vehicle_transform.get_forward_vector()
    
    # Araçtan engele doğru vektörü hesapla
    to_obstacle = carla.Vector3D(
        obstacle_location.x - vehicle_location.x,
        obstacle_location.y - vehicle_location.y,
        obstacle_location.z - vehicle_location.z
    )
    
    # Vektörleri normalleştir
    to_obstacle_length = math.sqrt(to_obstacle.x**2 + to_obstacle.y**2 + to_obstacle.z**2)
    if to_obstacle_length < 0.001:  # Çok yakınsa
        return True
    
    to_obstacle_normalized = carla.Vector3D(
        to_obstacle.x / to_obstacle_length,
        to_obstacle.y / to_obstacle_length,
        to_obstacle.z / to_obstacle_length
    )
    
    # İki vektör arasındaki açıyı bul (dot product kullanarak)
    dot_product = vehicle_forward.x * to_obstacle_normalized.x + \
                  vehicle_forward.y * to_obstacle_normalized.y + \
                  vehicle_forward.z * to_obstacle_normalized.z
    
    # dot_product'un değerine göre engelin aracın önünde olup olmadığına karar ver
    # 0'dan büyükse önde, değilse arkada veya yanda
    # Daha spesifik olarak, 0.3'ten büyükse yaklaşık 70 derece koni içinde demektir
    return dot_product > 0.3  # Yaklaşık 70 derecelik bir açıyı temsil ederimport time

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

    # Sadece x ve y koordinatlarını kullan (düzlemde hareket)
    target_vector_2d = np.array([target_vector.x, target_vector.y])
    current_vector_2d = np.array([current_forward_vector.x, current_forward_vector.y])
    
    # Vektörleri normalleştir
    if np.linalg.norm(target_vector_2d) > 0:
        target_vector_2d = target_vector_2d / np.linalg.norm(target_vector_2d)
    if np.linalg.norm(current_vector_2d) > 0:
        current_vector_2d = current_vector_2d / np.linalg.norm(current_vector_2d)
    
    # Çapraz çarpım ve nokta çarpımı ile açıyı hesapla
    cross_product = np.cross(current_vector_2d, target_vector_2d)
    dot_product = np.dot(current_vector_2d, target_vector_2d)
    
    # Arctan2 ile açıyı hesapla (-pi ile pi arasında)
    angle = np.degrees(np.arctan2(cross_product, dot_product))
    return angle


def predict_vehicle_position(current_location, velocity, seconds_ahead=1.0):
    """Aracın gelecekteki konumunu tahmin et."""
    future_x = current_location.x + velocity.x * seconds_ahead
    future_y = current_location.y + velocity.y * seconds_ahead
    future_z = current_location.z + velocity.z * seconds_ahead
    return carla.Location(x=future_x, y=future_y, z=future_z)


def calculate_safe_distance(speed, reaction_time=1.0, deceleration=5.0):
    """Güvenli mesafeyi hıza göre hesapla (Stopping Distance + Reaction Distance)."""
    # m/s cinsinden hız kullanılıyor
    reaction_distance = speed * reaction_time  # Reaksiyon mesafesi
    braking_distance = (speed**2) / (2 * deceleration)  # Durma mesafesi
    return reaction_distance + braking_distance


def calculate_safe_speed(distance_to_obstacle, min_safe_distance=2.0, max_speed=16.6):
    """Engele olan mesafeye göre güvenli hızı hesapla."""
    if distance_to_obstacle <= min_safe_distance:
        return 0.0  # Tamamen dur
    
    # Mesafe arttıkça hızı artır, ama maksimum hızı aşma
    safe_speed = (distance_to_obstacle - min_safe_distance) * 2.0
    return min(safe_speed, max_speed)


def check_for_obstacles_in_path(obstacle_sensor_data, path, vehicle_width=2.0):
    """Yol üzerinde engel olup olmadığını kontrol et."""
    if not obstacle_sensor_data:
        return None, float('inf')
    
    # En yakın engeli bul
    closest_obstacle = None
    min_distance = float('inf')
    
    for obstacle_id, obstacle_data in obstacle_sensor_data.items():
        obstacle_location = obstacle_data["location"]
        obstacle_distance = obstacle_data["distance"]
        
        # Engelin yolumuzda olup olmadığını kontrol et
        if is_obstacle_in_path(obstacle_location, path, vehicle_width) and obstacle_distance < min_distance:
            closest_obstacle = obstacle_id
            min_distance = obstacle_distance
    
    return closest_obstacle, min_distance


def is_obstacle_in_path(obstacle_location, path, vehicle_width=2.0):
    """Engelin yol üzerinde olup olmadığını kontrol et."""
    # Basit kontrol: Engel, belli bir mesafede yol üzerinde mi?
    if not path or len(path) < 2:
        return False
    
    for i in range(len(path) - 1):
        # Yol segmentini tanımla
        segment_start = path[i]
        segment_end = path[i + 1]
        
        # Engelin segmente olan en yakın noktasını bul
        closest_point = get_closest_point_on_segment(obstacle_location, segment_start, segment_end)
        
        # Engelin bu noktaya olan mesafesini hesapla
        distance = obstacle_location.distance(closest_point)
        
        # Eğer mesafe aracın genişliğinin yarısından azsa, engel yolda demektir
        if distance < vehicle_width / 2:
            return True
    
    return False


def get_closest_point_on_segment(point, segment_start, segment_end):
    """Bir noktanın bir çizgi segmentine olan en yakın noktasını bul."""
    segment_vector = carla.Vector3D(
        segment_end.x - segment_start.x,
        segment_end.y - segment_start.y,
        segment_end.z - segment_start.z
    )
    
    point_vector = carla.Vector3D(
        point.x - segment_start.x,
        point.y - segment_start.y,
        point.z - segment_start.z
    )
    
    segment_length_squared = segment_vector.x**2 + segment_vector.y**2 + segment_vector.z**2
    
    # Eğer segment bir noktaysa
    if segment_length_squared < 0.0001:
        return segment_start
    
    # Nokta vektörünün segment vektörü üzerindeki projeksiyonunu bul
    t = max(0, min(1, (point_vector.x * segment_vector.x + 
                         point_vector.y * segment_vector.y + 
                         point_vector.z * segment_vector.z) / segment_length_squared))
    
    # En yakın noktayı hesapla
    closest_x = segment_start.x + t * segment_vector.x
    closest_y = segment_start.y + t * segment_vector.y
    closest_z = segment_start.z + t * segment_vector.z
    
    return carla.Location(x=closest_x, y=closest_y, z=closest_z)


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
    """Yolu görselleştir."""
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


def main():
    global vehicle, camera, display, clock, path, path_index, obstacle_distance, obstacle_actor, debug

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # Bağlantı zaman aşımını artır
    world = client.get_world()
    map = world.get_map()
    debug = world.debug
    
    # Kontrol sistemleri
    throttle_controller = ThrottleController()
    brake_controller = BrakeController()
    steering_controller = SteeringController()
    
    # Engel sensörü verilerini depolamak için sözlük
    obstacle_data = {}

    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = map.get_spawn_points()

        if len(spawn_points) < 4:
            print("Yeterli spawnpoint bulunamadı. Çıkılıyor.")
            return
        
        start_waypoint = map.get_waypoint(spawn_points[0].location)
        end_waypoint = map.get_waypoint(spawn_points[3].location)

        if not start_waypoint or not end_waypoint:
            print("Geçersiz waypoint. Çıkılıyor.")
            return
        
        # Araç oluştur
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f"Araç oluşturuldu: {vehicle.id}")
        
        # A* algoritması ile yol planla
        path = astar(start_waypoint, end_waypoint.transform.location) 
        if path is None:
            print("Yol bulunamadı. Çıkılıyor.")
            return
        
        print(f"Yol bulundu: {len(path)} waypoint")

        # Kamera sensörü oluştur
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=-5, y=0, z=2.5), carla.Rotation(pitch=-15))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print("Kamera sensörü oluşturuldu")

        # Engel sensörü oluştur
        obstacle_bp = bp_lib.find('sensor.other.obstacle')
        obstacle_bp.set_attribute('distance', '5.0')  # Maksimum algılama mesafesini 25m'ye sınırla
        obstacle_bp.set_attribute('sensor_tick', '0.05')  # Sensör güncelleme sıklığı
        obstacle_bp.set_attribute('hit_radius', '0.5')  # Çarpışma yarıçapı
        obstacle_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
        obstacle_sensor = world.spawn_actor(obstacle_bp, obstacle_transform, attach_to=vehicle)
        print("Engel sensörü oluşturuldu")

        def obstacle_callback(event):
            """Engel tespit edildiğinde çağrılır."""
            global obstacle_distance, obstacle_actor
            if event.other_actor is not None:
                # Engel verilerini güncelle
                obstacle_id = event.other_actor.id
                obstacle_location = event.other_actor.get_location()
                
                # Aracın dönüş yönüne göre engelin önde olup olmadığını kontrol et
                is_in_front = is_obstacle_in_front(vehicle, obstacle_location)
                
                # Sadece önümüzdeki engelleri dikkate al
                if is_in_front:
                    obstacle_data[obstacle_id] = {
                        "actor": event.other_actor,
                        "distance": event.distance,
                        "location": obstacle_location
                    }
                    
                    # En yakın engeli bul
                    min_dist = float('inf')
                    closest_id = None
                    
                    for obj_id, obj_data in obstacle_data.items():
                        if obj_data["distance"] < min_dist and is_obstacle_in_front(vehicle, obj_data["location"]):
                            min_dist = obj_data["distance"]
                            closest_id = obj_id
                    
                    if closest_id is not None:
                        obstacle_distance = obstacle_data[closest_id]["distance"]
                        obstacle_actor = obstacle_data[closest_id]["actor"]
                        
                        # Engel çok uzakta ise dikkate alma (örn. 20m'den fazla)
                        if obstacle_distance > 5.0:
                            obstacle_distance = None
                            obstacle_actor = None
                            print(f"Engel tespit edildi fakat çok uzakta: {event.distance:.2f}m")
                            return
                        
                        vehicle_speed = get_speed(vehicle)
                        safe_distance = calculate_safe_distance(vehicle_speed)
                        
                        print(f"Engel tespit edildi: {obstacle_actor.type_id}, mesafe: {obstacle_distance:.2f}m, güvenli mesafe: {safe_distance:.2f}m")
                        
                        # Engel yakınsa ve tehlike oluşturuyorsa ekstra uyarı ver
                        if obstacle_distance < safe_distance:
                            print(f"DİKKAT! Güvenli mesafenin altında! Frenleme gerekli!")
                else:
                    print(f"Engel araç arkasında, dikkate alınmıyor: {event.other_actor.type_id}")
        
        # Sensörü dinle
        obstacle_sensor.listen(obstacle_callback)

        # Pygame başlat
        pygame.init()
        display = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("CARLA Otopilot Simülasyonu")
        clock = pygame.time.Clock()

        # Kamera görüntüsünü işle
        def process_image(image):
            surface = carla_image_to_pygame_surface(image)
            display.blit(surface, (0, 0))
            
            # Ekstra bilgileri ekrana yaz
            font = pygame.font.SysFont('Arial', 20)
            current_speed = get_speed(vehicle)
            speed_text = font.render(f"Hız: {current_speed * 3.6:.1f} km/s", True, (255, 255, 255))
            display.blit(speed_text, (10, 10))
            
            if obstacle_distance is not None and obstacle_actor is not None:
                # Engel bilgilerini göster
                color = (255, 50, 50) if obstacle_distance < calculate_safe_distance(current_speed) else (255, 255, 100)
                obstacle_text = font.render(f"Engel: {obstacle_actor.type_id.split('.')[-1]}, Mesafe: {obstacle_distance:.2f} m", True, color)
                display.blit(obstacle_text, (10, 40))
                
                # Güvenli mesafeyi hesapla ve göster
                safe_distance = calculate_safe_distance(current_speed)
                safe_text = font.render(f"Güvenli Mesafe: {safe_distance:.2f} m", True, 
                                      (100, 255, 100) if obstacle_distance > safe_distance else (255, 50, 50))
                display.blit(safe_text, (10, 70))
                
                # Durum bilgisi göster
                if obstacle_distance < safe_distance * 0.5:
                    status_text = font.render("DURUM: ACİL FREN!", True, (255, 0, 0))
                elif obstacle_distance < safe_distance:
                    status_text = font.render("DURUM: YAVAŞLAMA", True, (255, 165, 0))
                else:
                    status_text = font.render("DURUM: GÜVENLİ SÜRÜŞ", True, (0, 255, 0))
                display.blit(status_text, (10, 100))
            else:
                status_text = font.render("DURUM: ENGEL YOK", True, (0, 255, 0))
                display.blit(status_text, (10, 40))
                
            # Ekranı güncelle
            pygame.display.flip()

        # Kamerayı dinle
        camera.listen(process_image)

        # Yolu görselleştir
        draw_waypoints(world, path)

        # Hedef hız (30 km/h = 8.33 m/s)
        target_speed = 8.33  
        max_speed = 9.0
        last_time = time.time()
        
        # Sürüş fonksiyonu
        while True:
            # Pygame olaylarını kontrol et
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            # Zaman hesaplama
            now = time.time()
            dt = now - last_time
            last_time = now
            
            # Mevcut araç durumu
            current_location = vehicle.get_location()
            current_velocity = vehicle.get_velocity()
            current_speed = get_speed(vehicle)
            
            # Engel durumuna göre hız ayarla
            safe_speed = max_speed
            emergency_brake = False
            
            if obstacle_distance is not None:
                # Güvenli mesafeyi hesapla
                safe_distance = calculate_safe_distance(current_speed)
                
                # Eğer engel güvenli mesafeden yakınsa
                if obstacle_distance < safe_distance:
                    # Mesafeye göre güvenli hız hesapla
                    safe_speed = calculate_safe_speed(obstacle_distance)
                    
                    # Eğer çok yakınsa acil fren yap
                    if obstacle_distance < safe_distance * 0.5:
                        emergency_brake = True
                        print("ACİL FREN!")
            
            # Hedef hızı ayarla (normal durum veya engel varsa)
            target_speed = min(safe_speed, max_speed)
            
            # Yol takibi
            if path and path_index < len(path):
                target_location = path[path_index]
                distance_to_target = current_location.distance(target_location)
                
                # Hedefe yeterince yaklaştıysa bir sonraki waypointe geç
                if distance_to_target < max(2.0, current_speed * 0.5):
                    path_index += 1
                    # Yeni hedefi göster
                    if path_index < len(path):
                        debug.draw_point(path[path_index], size=0.1, color=carla.Color(255, 0, 0), life_time=0.5)
                    continue
                
                # Direksiyon açısını hesapla
                target_angle = calculate_steering_angle(current_location, target_location)
                current_angle = 0  # Anlık açı
                steer = steering_controller.run_step(current_angle, target_angle, dt)
                
                # Gaz ve fren kontrolü
                throttle = 0.0
                brake = 0.0
                
                if emergency_brake:
                    # Acil durum freni
                    throttle = 0.0
                    brake = 1.0
                else:
                    # Normal sürüş kontrolü
                    if current_speed > target_speed:
                        # Hız fazla ise frenle
                        brake = brake_controller.run_step(target_speed, current_speed, dt)
                        throttle = 0.0
                    else:
                        # Hız az ise hızlan
                        brake = 0.0
                        throttle = throttle_controller.run_step(target_speed, current_speed, dt)
                
                # Kontrolü uygula
                control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
                vehicle.apply_control(control)
                
                # Debugging bilgilerini göster
                if debug:
                    # Hedef noktayı göster
                    debug.draw_point(target_location, size=0.1, color=carla.Color(0, 0, 255), life_time=0.1)
                    
                    # Aracın gelecekteki konumunu tahmin et ve göster
                    future_location = predict_vehicle_position(current_location, current_velocity)
                    debug.draw_point(future_location, size=0.1, color=carla.Color(255, 255, 0), life_time=0.1)
                    
                    # Engel varsa göster
                    if obstacle_actor:
                        debug.draw_line(
                            current_location,
                            obstacle_actor.get_location(),
                            thickness=0.1,
                            color=carla.Color(255, 0, 0) if emergency_brake else carla.Color(255, 100, 0),
                            life_time=0.1
                        )
            else:
                # Hedefe ulaşıldı, dur
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                print("Hedefe ulaşıldı.")
                time.sleep(2)  # Biraz bekle
                break

            # Düşük FPS'te tick
            world.tick()
            clock.tick(30)

    except Exception as e:
        print(f"Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Temizleniyor...")
        # Tüm aktörleri temizle
        if vehicle:
            vehicle.destroy()
        if camera:
            camera.destroy()
        if 'obstacle_sensor' in locals():
            obstacle_sensor.destroy()
        pygame.quit()


def get_speed(vehicle):
    """Aracın hızını m/s cinsinden hesapla."""
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


if __name__ == '__main__':
    main()

