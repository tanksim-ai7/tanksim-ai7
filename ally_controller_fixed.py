from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
from move.risk_planner import RiskDStarPlanner as DStarLitePlanner
from move.dstar_lite_planner_cost import ObstacleRect as DStarLiteObstacleRect
import numpy as np
import math
import matplotlib
from move.pid_controller import (
    update_info_speed,
    read_player_speed_kmh,
    read_player_body_yaw_deg,
    get_control_dt,
    check_destination_change,
    update_arrival_state,
    calculate_target_speed_kmh,
    find_upcoming_corner,
    calculate_corner_speed_limit,
    calculate_steering_command,
    apply_alignment_speed_limit,
    make_longitudinal_command,
    make_stop_command,
)
matplotlib.use("Agg")

app = Flask(__name__)
model = YOLO('yolov8n.pt')
print(model.names)

path_flag = False # 경로 최초 탐색 여부
path = []
path_idx = 0 # path를 위한 idx
all_info = None # info 정보
dest = None # 목적지

path_planner = DStarLitePlanner() # 초기화할 때 고도정보도 같이 넣어준다.

def update_dstar_obstacles_from_payload(payload: dict):
    obs_list = []
    for item in payload.get("obstacles", []):
        obs = DStarLiteObstacleRect.from_min_max(
            x_min=item["x_min"],
            x_max=item["x_max"],
            z_min=item["z_min"],
            z_max=item["z_max"],
        )
        obs_list.append(obs)
    path_planner.set_obstacles(obs_list)

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()
    # print(results[0].boxes.data)
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
    all_info = data

    player_pos = data.get("playerPos", {})

    if (
        player_pos.get("x") is not None
        and player_pos.get("z") is not None
    ):
        update_info_speed(
            data,
            [
                player_pos["x"],
                player_pos["z"],
            ],
        )

    return jsonify({"status": "success", "control": ""})

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

    # print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    # print(f"🎯 Turret received: x={turret_x}, y={turret_y}")

    if path_idx >= len(path):
        return jsonify({"moveAD": {"command": "", "weight": 0}, "moveWS": {"command": "", "weight": 0}})
    if dest is None:
        return jsonify(make_stop_command())
    current_position = [
        float(pos_x),
        float(pos_z),
    ]
    current_speed_kmh = read_player_speed_kmh()
    if current_speed_kmh is None:
        return jsonify(make_stop_command())
    body_yaw_deg = read_player_body_yaw_deg(
        all_info or {}
    )

    # yaw 값이 없으면 좌/우 어느 방향으로 회전할지 계산할 수 없다.
    if body_yaw_deg is None:
        return jsonify(make_stop_command())

    # --------------------------------------------------------
    # 4. 목적지 변경 확인
    # --------------------------------------------------------
    # ally-controller의 dest는 (x, y, z) 형태이지만
    # PID는 X-Z 평면만 사용하므로 (x, z)만 전달한다.
    pid_destination = (
        float(dest[0]),
        float(dest[2]),
    )

    # 목적지가 변경됐으면 이전 목적지에서 누적된
    # PID I/D 상태, 도착 latch, 최종 제동 상태를 초기화한다.
    check_destination_change(
        pid_destination
    )

    # --------------------------------------------------------
    # 5. PID 제어 주기 dt
    # --------------------------------------------------------
    # now:
    #   목적지 마지막 제동시간 계산에 사용.
    #
    # dt:
    #   speed PID / steering PID의 I항과 D항 계산에 사용.
    now, dt = get_control_dt()

    # --------------------------------------------------------
    # 6. 현재 위치 -> 최종 목적지 거리
    # --------------------------------------------------------
    goal_dx = pid_destination[0] - current_position[0]
    goal_dz = pid_destination[1] - current_position[1]

    distance_to_goal = math.hypot(
        goal_dx,
        goal_dz,
    )

    # --------------------------------------------------------
    # 7. 목적지 도착 / 마지막 제동
    # --------------------------------------------------------
    # STOP_DISTANCE_M 안에 들어오면 arrival_latched=True가 되고,
    # 일정 시간만 S 제동 후 입력을 해제한다.
    arrival_command = update_arrival_state(
        distance_to_goal=distance_to_goal,
        current_speed_kmh=current_speed_kmh,
        now=now,
    )

    # 목적지 도착 상태라면 일반 PID 계산을 하지 않고
    # 마지막 제동/정지 명령을 바로 반환한다.
    if arrival_command is not None:
        return jsonify(arrival_command)

    # --------------------------------------------------------
    # 8. 기존 D* Lite path를 따라 A/D 조향 계산
    # --------------------------------------------------------
    # 여기서는 새 경로를 계산하지 않는다.
    #
    # path는 기존 ally-controller의
    # /init 또는 /update_obstacle에서 path_planner.find_path()로
    # 이미 계산된 경로를 그대로 사용한다.
    steering_command, steering_info = calculate_steering_command(
        current_position=current_position,
        body_yaw_deg=body_yaw_deg,
        path=path,
        current_speed_kmh=current_speed_kmh,
        dt=dt,
    )

    # Look-ahead target을 만들 수 없는 경우 안전하게 정지한다.
    if steering_info is None:
        return jsonify(make_stop_command())

    # --------------------------------------------------------
    # 9. 현재 경로 앞쪽 코너 탐색
    # --------------------------------------------------------
    upcoming_corner = find_upcoming_corner(
        path=path,
        current_position=current_position,
    )

    # Sharp / Medium / Gentle 코너의 각도와 거리에 따라
    # 코너 진입 전 허용 목표속도를 계산한다.
    corner_speed_limit_kmh = calculate_corner_speed_limit(
        upcoming_corner
    )

    # --------------------------------------------------------
    # 10. 목적지까지 남은 거리 기준 목표속도
    # --------------------------------------------------------
    # v^2 = 2ad 기반으로 목적지에서 정지 가능한 목표속도를 계산한다.
    destination_speed_limit_kmh = calculate_target_speed_kmh(
        distance_to_goal
    )

    # 목적지 감속 제한과 코너 감속 제한 중
    # 더 낮은 속도를 실제 PID 목표속도로 사용한다.
    target_speed_kmh = min(
        destination_speed_limit_kmh,
        corner_speed_limit_kmh,
    )

    # --------------------------------------------------------
    # 11. 속도 PID
    # --------------------------------------------------------
    # PID error = 목표속도 - 현재속도
    speed_error_kmh = (
        target_speed_kmh
        - current_speed_kmh
    )

    # PID 출력 범위:
    #   + 값 -> W
    #   - 값 -> S
    pid_output = speed_pid.update(
        speed_error_kmh,
        dt,
    )

    # PID 숫자 출력을 실제 시뮬레이터 moveWS 명령으로 변환한다.
    command = make_longitudinal_command(
        pid_output
    )

    # --------------------------------------------------------
    # 12. 차체 정렬 상태에 따른 전진 출력 제한
    # --------------------------------------------------------
    # 경로와 차체 방향이 크게 어긋난 상태에서
    # W=1.0으로 바로 가속하는 것을 방지한다.
    command = apply_alignment_speed_limit(
        command=command,
        heading_error_deg=steering_info[
            "heading_error_deg"
        ],
        current_speed_kmh=current_speed_kmh,
    )

    # --------------------------------------------------------
    # 13. A/D 조향 명령 결합
    # --------------------------------------------------------
    # make_longitudinal_command()가 만든 moveWS에
    # calculate_steering_command()가 계산한 moveAD를 합친다.
    command["moveAD"] = steering_command



    # print("🔁 Sent Combined Action:", command)
    return jsonify({"moveAD": {"command": "", "weight": 0}, "moveWS": {"command": "", "weight": 0}})

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

        global dest
        dest = (x, y, z)

        # print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    update_dstar_obstacles_from_payload(data)

    if path_flag:
        global path, path_idx
        path = [] # path 초기화
        path_idx = 0 # path idx 초기화
        path = path_planner.find_path((all_info['playerPos']['x'], all_info['playerPos']['z']), (dest[0], dest[2]))
        path_planner.plot(path, save_path='terrain_map')

    # print("🪨 Obstacle Data:", data)
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
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 60,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 27.23,
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
    global path, path_idx, path_flag

    path_planner.set_risk_layers()
    path = path_planner.find_path((60, 27.23), (dest[0], dest[2]))
    path_planner.plot(path, save_path='terrain_map')

    path_idx = 0
    path_flag = True
    
    # print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    # print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
