from flask import Flask, request, jsonify
import os

from move.risk_planner import RiskDStarPlanner as DStarLitePlanner
from move.pid_controller import TankDriveController
from fire.fire_module import FireModule
import requests
import threading

import evade_steering as evd

import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

path_planner = DStarLitePlanner(is_enemy=True)
drive_controller = TankDriveController(path_planner, 'dstar_enemy_map.png')

fm = FireModule()

NEXT_ALLY_DEST = None
SEQ_FLAG = 'first'
ENEMY_DEST_LIST = [(111.0, 172.0)]
ENEMY_DEST_IDX = 0

@app.route('/init', methods=['POST'])
def init():
    """episode 설정을 반환하고 TankDriveController 상태를 초기화한다."""
    # 기존 통합 서버의 simulator config를 유지한다.
    config = {
        "startMode": "start",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 111,
        "rdStartY": 15,
        "rdStartZ": 172,
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

    drive_controller.initialize(start_position=(111.0, 172.0))

    return jsonify({"status": "success"}), 200

@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)

    playerPos = data['playerPos']
    playerSpeed = data['playerSpeed']
    playerTurretX = data['playerTurretX']
    playerTurretY = data['playerTurretY']
    playerBodyX = data['playerBodyX']
    playerBodyY = data['playerBodyY']
    playerBodyZ = data['playerBodyZ']
    playerHealth = data['playerHealth']

    data['playerPos'] = data['enemyPos']
    data['playerSpeed'] = data['enemySpeed']
    data['playerTurretX'] = data['enemyTurretX']
    data['playerTurretY'] = data['enemyTurretY']
    data['playerBodyX'] = data['enemyBodyX']
    data['playerBodyY'] = data['enemyBodyY']
    data['playerBodyZ'] = data['enemyBodyZ']
    data['playerHealth'] = data['enemyHealth']

    data['enemyPos'] = playerPos
    data['enemySpeed'] = playerSpeed
    data['enemyTurretX'] = playerTurretX
    data['enemyTurretY'] = playerTurretY
    data['enemyBodyX'] = playerBodyX
    data['enemyBodyY'] = playerBodyY
    data['enemyBodyZ'] = playerBodyZ
    data['enemyHealth'] = playerHealth

    global SEQ_FLAG, ENEMY_DEST_IDX, ENEMY_DEST_LIST
    if SEQ_FLAG == 'first':
        if data['enemyPos']['z'] > 180.0 and ENEMY_DEST_IDX == 0:
            ENEMY_DEST_LIST.append((131.0, 250.0))
            ENEMY_DEST_IDX += 1
            dest = {
                "destination": f"{ENEMY_DEST_LIST[ENEMY_DEST_IDX][0]}, {data['playerPos']['y']}, {ENEMY_DEST_LIST[ENEMY_DEST_IDX][1]}"
            }
            ENEMY_DEST_IDX += 1
            response, status = drive_controller.handle_set_destination(dest)

        if 130.0 <= data['playerPos']['x'] <= 132.0 and 249.0 <= data['playerPos']['z'] <= 251.0:
            SEQ_FLAG = 'second'

    if SEQ_FLAG == 'second':
        if ENEMY_DEST_IDX <= len(ENEMY_DEST_LIST):
            if ENEMY_DEST_LIST[ENEMY_DEST_IDX-1][0]-1 <= data['playerPos']['x'] <= ENEMY_DEST_LIST[ENEMY_DEST_IDX-1][0]+1 and\
               ENEMY_DEST_LIST[ENEMY_DEST_IDX-1][1]-1 <= data['playerPos']['z'] <= ENEMY_DEST_LIST[ENEMY_DEST_IDX-1][1]+1:

                ENEMY_DEST_LIST.append(path_planner.get_random_destination(data, NEXT_ALLY_DEST))
                dest = {
                    "destination": f"{ENEMY_DEST_LIST[ENEMY_DEST_IDX][0]}, {data['playerPos']['y']}, {ENEMY_DEST_LIST[ENEMY_DEST_IDX][1]}"
                }
                ENEMY_DEST_IDX += 1

                response, status = drive_controller.handle_set_destination(dest)

    return jsonify(response)

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    response, status = drive_controller.handle_update_obstacles(request.get_json())

    return jsonify(response), status

@app.route('/get_next_dest', methods=['POST'])
def get_next_dest():
    data = request.get_json(force=True)

    global NEXT_ALLY_DEST
    NEXT_ALLY_DEST = data['next_dest']

@app.route('/get_action', methods=['POST'])
def get_action():
    """
    주행 명령과 사격 명령을 독립 계산한 뒤 하나로 합친다.

    TankDriveController 소유:
        moveWS, moveAD

    FireModule 소유:
        turretQE, turretRF, fire
    """
    # 이번 /get_action JSON snapshot.
    data = request.get_json(force=True)

    if SEQ_FLAG == 'first':
        # D* Lite + PID 차체 이동/조향 명령.
        rst_cmd = drive_controller.get_action(data)
    elif SEQ_FLAG == 'second':
        # 선회
        rst_cmd = drive_controller.get_action(data)
        fire_inputs = (
            drive_controller.get_fire_control_inputs(
                rst_cmd
            )
        )

        turret_cmd = fm.get_turret_command(
            my_vel=fire_inputs["my_vel"],
            body_rate_dps=fire_inputs["body_rate_dps"],
            hull_settled=fire_inputs["hull_settled"],
        )

        rst_cmd["turretQE"] = turret_cmd["turretQE"]
        rst_cmd["turretRF"] = turret_cmd["turretRF"]
        rst_cmd["fire"] = turret_cmd["fire"]
    else:
        # 두번 포격 당하고 STOP
        rst_cmd = {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False
        }

    return jsonify(rst_cmd)

@app.route('/go_third_step', methods=['POST'])
def go_third_step():
    global SEQ_FLAG
    SEQ_FLAG = 'third'

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    fm.on_impact(data)

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)
