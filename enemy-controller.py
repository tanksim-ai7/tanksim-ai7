from flask import Flask, request, jsonify
import os

from move.risk_planner import RiskDStarPlanner as DStarLitePlanner
from move.pid_controller import TankDriveController


app = Flask(__name__)

path_planner = DStarLitePlanner(is_enemy=True)
drive_controller = TankDriveController(path_planner, "dstar_enemy_map.png")


tmp_flag = True

@app.route('/init', methods=['GET'])
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

    return True

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
    global tmp_flag
    if tmp_flag:
        tmp_flag = False
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
