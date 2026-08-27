from flask import Flask, request, jsonify
import os

from move.risk_planner import RiskDStarPlanner as DStarLitePlanner
from move.pid_controller import TankDriveController
import requests
import threading

app = Flask(__name__)

path_planner = DStarLitePlanner(is_enemy=True)
drive_controller = TankDriveController(path_planner, "dstar_enemy_map.png")

# (59.0, 280.0) => 첫 시작 위치
# 나머지 => 순서대로 진행될 목적지 좌표
ENEMY_DEST_LIST = [(59.0, 280.0), (60.0, 5.23), (285.0, 285.0)]
ENEMY_DEST_IDX = 0

tmp_flag = True

@app.route('/init', methods=['POST'])
def init():
    """episode 설정을 반환하고 TankDriveController 상태를 초기화한다."""
    # 기존 통합 서버의 simulator config를 유지한다.
    config = {
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59,
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": True,
        "detectMode": False,
        "logMode": True,
        "stereoCameraMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit": True,
    }

    drive_controller.initialize(start_position=(59.0, 280.0))

    return jsonify({"status": "success"}), 200

@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    
    dest = {
        "destination": f"{data['playerPos']['x']}, {data['playerPos']['y']}, {data['playerPos']['z']}"
    }
    data['playerPos'] = data['enemyPos']
    data['playerSpeed'] = data['enemySpeed']
    data['playerTurretX'] = data['enemyTurretX']
    data['playerTurretY'] = data['enemyTurretY']
    data['playerBodyX'] = data['enemyBodyX']
    data['playerBodyY'] = data['enemyBodyY']
    data['playerBodyZ'] = data['enemyBodyZ']

    response, status = drive_controller.handle_info(data)
    global ENEMY_DEST_IDX
    if ENEMY_DEST_IDX < len(ENEMY_DEST_LIST)-1:
        if ENEMY_DEST_LIST[ENEMY_DEST_IDX][0]-1 <= data['playerPos']['x'] <=  ENEMY_DEST_LIST[ENEMY_DEST_IDX][0]+1 and\
           ENEMY_DEST_LIST[ENEMY_DEST_IDX][1]-1 <= data['playerPos']['z'] <=  ENEMY_DEST_LIST[ENEMY_DEST_IDX][1]+1:

                ENEMY_DEST_IDX += 1
                dest = {
                    "destination": f"{ENEMY_DEST_LIST[ENEMY_DEST_IDX][0]}, {data['playerPos']['y']}, {ENEMY_DEST_LIST[ENEMY_DEST_IDX][1]}"
                }
                response, status = drive_controller.handle_set_destination(dest)

    return jsonify(response)

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    response, status = drive_controller.handle_update_obstacles(request.get_json())

    return jsonify(response), status

@app.route('/get_action', methods=['POST'])
def get_action():
    data = request.get_json(force=True)
    rst_cmd = drive_controller.get_action(data)
    
    def send_path_background():
        try:
            enemy_path = getattr(drive_controller.planner, "last_path", [])
            requests.post("http://127.0.0.0:5000/get_enemy_path", json={"enemy_path": enemy_path}, timeout=0.02)
        except:
            pass

    threading.Thread(target=send_path_background, daemon=True).start()
    
    return jsonify(rst_cmd)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)
