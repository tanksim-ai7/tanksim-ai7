# library선언 및 변수 선언
from flask import Flask, request, jsonify
from ultralytics import YOLO
from pathlib import Path

import os
import torch
import math
import uuid
import cv2
import numpy as np

app = Flask(__name__)
MODEL_PATH = Path("./models/260819_best.pt")
model = YOLO(str(MODEL_PATH))

target_classes = {
    0: 'Car', 
    1: 'House', 
    2: 'Human1', 
    3: 'Human2', 
    4: 'Human3', 
    5: 'Mine', 
    6: 'Rock', 
    7: 'Tank1', 
    8: 'Tank2', 
    9: 'Tent', 
    10: 'Tree', 
    11: 'Wall'
}