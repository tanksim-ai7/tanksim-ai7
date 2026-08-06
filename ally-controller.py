from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import math

from move.dstar_lite_planner import DStarLitePlanner, ObstacleRect

app = Flask(__name__)
model = YOLO('yolov8n.pt')

all_info = None # info 정보
idx = 0 # 경로 point의 idx
path = None # 경로에 대한 list
dest = None # 목적지에 대한 변수
path_flag = False # path finding 최초 여부 flag

planner = DStarLitePlanner(
    grid_min_x=0.0,
    grid_max_x=300.0,
    grid_min_z=0.0,
    grid_max_z=300.0,
    cell_size=1.0,
    obstacle_margin=2.0,
    allow_diagonal=True,
)

print(model.names)
combined_commands = [
    {
        "moveWS": {"command": "W", "weight": 1.0},
        "moveAD": {"command": "D", "weight": 1.0},
        "turretQE": {"command": "Q", "weight": 0.7},
        "turretRF": {"command": "R", "weight": 0.5},
        "fire": False
    },
    {
        "moveWS": {"command": "W", "weight": 0.6},
        "moveAD": {"command": "A", "weight": 0.4},
        "turretQE": {"command": "E", "weight": 0.8},
        "turretRF": {"command": "R", "weight": 0.3},
        "fire": True
    },
    {
        "moveWS": {"command": "W", "weight": 0.5},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "E", "weight": 0.4},
        "turretRF": {"command": "R", "weight": 0.6},
        "fire": False
    },
    {
        "moveWS": {"command": "W", "weight": 0.3},
        "moveAD": {"command": "D", "weight": 0.3},
        "turretQE": {"command": "E", "weight": 0.5},
        "turretRF": {"command": "R", "weight": 0.7},
        "fire": True
    },
    {
        "moveWS": {"command": "W", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "E", "weight": 0.5},
        "turretRF": {"command": "R", "weight": 0.5},
        "fire": False
    },
    {
        "moveWS": {"command": "W", "weight": 0.8},
        "moveAD": {"command": "A", "weight": 0.6},
        "turretQE": {"command": "E", "weight": 0.9},
        "turretRF": {"command": "R", "weight": 0.2},
        "fire": True
    },
    {
        "moveWS": {"command": "W", "weight": 1.0},
        "moveAD": {"command": "D", "weight": 1.0},
        "turretQE": {"command": "E", "weight": 1.0},
        "turretRF": {"command": "R", "weight": 1.0},
        "fire": True
    },
    {
        "moveWS": {"command": "W", "weight": 0.2},
        "moveAD": {"command": "A", "weight": 0.9},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "R", "weight": 0.9},
        "fire": False
    },
    {
        "moveWS": {"command": "S", "weight": 0.4},
        "moveAD": {"command": "D", "weight": 0.4},
        "turretQE": {"command": "E", "weight": 0.6},
        "turretRF": {"command": "F", "weight": 0.6},
        "fire": True
    },
    {
        "moveWS": {"command": "W", "weight": 0.8},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "Q", "weight": 0.5},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False
    },
    {
        "moveWS": {"command": "STOP", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": True
    },
    {
        "moveWS": {"command": "S", "weight": 0.2},
        "moveAD": {"command": "A", "weight": 0.2},
        "turretQE": {"command": "E", "weight": 0.2},
        "turretRF": {"command": "F", "weight": 0.2},
        "fire": False
    }
]


@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()
    print(results[0].boxes.data)
    target_classes = {0: "tank",1: "rock", 2: "car", 7: "truck", 15: "rock"}
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': False
            })

    return jsonify(filtered_results)

@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    left_image = request.files.get('left_image')
    right_image = request.files.get('right_image')

    if not left_image or not right_image:
        return jsonify({"result": "error", "message": "Left or Right image missing"}), 400

    left_path = "temp_left.jpg"
    right_path = "temp_right.jpg"

    try:
        left_image.save(left_path)
        right_image.save(right_path)
    except Exception as e:
        return jsonify({"result": "error", "message": str(e)}), 500

    return jsonify({"result": "success"})
    
@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    global all_info
    all_info = data # info 정보 전역변수에 저장

    #print("📨 /info data received:", data)

    # Auto-pause after 15 seconds
    #if data.get("time", 0) > 15:
    #    return jsonify({"status": "success", "control": "pause"})
    # Auto-reset after 15 seconds
    #if data.get("time", 0) > 15:
    #    return jsonify({"stsaatus": "success", "control": "reset"})
    return jsonify({"status": "success", "control": ""})

def get_angle_diff(target, current):
    diff = (target - current + 180) % 360 - 180
    return diff

@app.route('/get_action', methods=['POST'])
def get_action():
    data = request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})

    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    degree = all_info['playerBodyX']
    now_speed = all_info['playerSpeed']

    # print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    # print(f"🎯 Turret received: x={turret_x}, y={turret_y}")
    if path:
        target = path[idx]

        dx = target[0] - pos_x
        dz = target[1] - pos_z

        distance = math.hypot(dx, dz) # 대각선 거리 -> 직선 거리

        target_angle = math.degrees(math.atan2(dx, dz))
        if target_angle < 0:
            target_angle += 360

        angle_diff = get_angle_diff(target_angle, degree)
        abs_angle_diff = abs(angle_diff)


        MAX_SPEED = 20.0 
    
        if abs_angle_diff > 15:
            target_speed = 0.05  
        else:
            target_speed = max(6, MAX_SPEED * (1 - (abs_angle_diff / 15.0)))


        if now_speed > 5.0:
            BRAKING_DISTANCE = 30.0
        else:
            BRAKING_DISTANCE = max(5.0, math.sqrt(max(0.0, now_speed)) * 4.5)

        if distance < BRAKING_DISTANCE:
            distance_ratio = max(0.0, min(1.0, distance / BRAKING_DISTANCE))
            braking_curve = distance_ratio * distance_ratio * distance_ratio * distance_ratio
            distance_target_speed = max(1.5, MAX_SPEED * braking_curve)
            target_speed = min(target_speed, distance_target_speed)



        forward_power = 0.0
        brake_power = 0.0
        ws_command = "W"

        if now_speed < target_speed:
            ws_command = "W"
            speed_diff = target_speed - now_speed
            forward_power = min(1.0, max(0.4, speed_diff / 1.5))
        else:
            if now_speed <= 3.0:
                ws_command = "W"
                forward_power = 0.0  
                brake_power = 0.0
            else:
                ws_command = "S"
                if now_speed > 15.0 and distance < BRAKING_DISTANCE:
                    brake_power = 1.0
                else:
                    speed_gap = now_speed - target_speed
                    brake_power = min(1.0, max(0.4, (speed_gap * speed_gap) / 1.0))

                if now_speed < 8.0:
                    brake_power = min(0.35, brake_power)



        speed_ratio_squared = (now_speed / MAX_SPEED) ** 2
        STOPPING_DISTANCE = max(5.5, speed_ratio_squared * 65.0)

        if distance < STOPPING_DISTANCE and now_speed > 0.5:
            if ws_command == "W":
                forward_power = 0.0
                
            ws_command = "S"
            
            # 목적지에 가까워질수록(ratio가 1.0에 가까워짐), 속도가 빠를수록 강하게 제동
            stop_distance_ratio = (STOPPING_DISTANCE - distance) / STOPPING_DISTANCE
            stop_speed_ratio = min(1.0, now_speed / MAX_SPEED)
            
            # 승수를 3.5로 상향하여 65m 영역 진입과 동시에 풀 브레이크(1.0)가 조기에 걸리도록 유도
            stopping_intensity = min(5.0, max(0.5, stop_distance_ratio * stop_speed_ratio * 15.0))
            brake_power = max(brake_power, stopping_intensity)



        speed_factor = max(0.35, 1.0 - (now_speed / 90.0))
        turn_power = min(0.5 * speed_factor, abs_angle_diff / 20.0)
        
        command = {}
        if angle_diff > 1.0:    
            command["moveAD"] = {"command": "D", "weight": turn_power}
        elif angle_diff < -1.0:
            command["moveAD"] = {"command": "A", "weight": turn_power}
        else:
            command["moveAD"] = {"command": "", "weight": 0}

        if ws_command == "W":
            command["moveWS"] = {"command": "W", "weight": forward_power}
        else:
            command["moveWS"] = {"command": "S", "weight": brake_power}



        angle_mitigation = max(0.15, 1.0 - (abs_angle_diff / 60.0))
        arrival_threshold = max(1.5, (now_speed * 0.2) * angle_mitigation) 
        
        if distance < arrival_threshold:
            idx += 1
    else: # path가 없을 경우
        command = {
            "moveWS": {"command": "STOP", "weight": 0.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False
        }

    # print("🔁 Sent Combined Action:", command)
    return jsonify(command)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")

        global dest
        dest = (x, y, z) # 목적지 저장

        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400


def update_obstacles_from_payload(payload: dict):
    """
    오브젝트 정보 planner class에 입력
    """
    obs_list = []
    for item in payload.get("obstacles", []):
        obs = ObstacleRect.from_min_max(
            x_min=item["x_min"],
            x_max=item["x_max"],
            z_min=item["z_min"],
            z_max=item["z_max"],
        )
        obs_list.append(obs)
    planner.set_obstacles(obs_list)

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    update_obstacles_from_payload(data)

    # 최초 탐색이 아닌 경우
    if path_flag:
        print('경로 재탐색')
        global path, idx
        idx = 0
        path = planner.find_path((all_info['playerPos']['x'], all_info['playerPos']['z']), (dest[0], dest[2]))
        planner.plot(path, show_grid=True, title="A* Demo (300x300)", fname='path')

    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST']) 
def collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return jsonify({'status': 'success', 'message': 'Collision data received'})

#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
    start_point = (60, 10, 27.23)
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": start_point[0],  #Blue Start Position
        "blStartY": start_point[1],
        "blStartZ": start_point[2],
        "rdStartX": 59, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": True,
        "detectMode": False,
        "logMode": False,
        "stereoCameraMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit" : True
    }

    global path, idx, path_flag
    planner.reset_planner() # 처음부터 path finding을 위한 초기화
    path = planner.find_path((start_point[0], start_point[2]), (dest[0], dest[2]))
    planner.plot(path, show_grid=True, title="A* Demo (300x300)", fname='path')

    idx = 0 # path point idx 초기화
    path_flag = True 

    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
