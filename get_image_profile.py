import cv2

import numpy as np
import math
from PIL import Image

# 画像を読み込む
image_path = r"images\a3_with_a4_parallelogram_blurred_600dpi.png"
image = cv2.imread(image_path)


# 画像のdpi情報を取得
def get_image_dpi(image_path):
    image = Image.open(image_path)
    dpi = image.info.get("dpi", (96, 96))  # デフォルト値を96 DPIに設定
    return dpi


dpi = get_image_dpi(image_path)
pixels_per_mm = dpi[0] * 25.4

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ガウシアンブラーを適用してノイズを低減
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # エッジ検出を使用
# edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
# cv2.imwrite("edges.png", edges)

# 輪郭を見つける
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 最大の輪郭を見つける（紙の輪郭であるはず）
contour = max(contours, key=cv2.contourArea)

# 輪郭を多角形に近似
epsilon = 0.02 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

# 多角形が4点を持つことを確認
if len(approx) != 4:
    raise ValueError("検出された輪郭は4点を持っていません。")

# 頂点の座標を取得
vertices = [tuple(point[0]) for point in approx]


# 辺の長さを計算
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 画像の解像度（ピクセル/メートル）を指定
pixels_per_meter = 3779.5275590551  # 96 DPI (dots per inch) * 39.3701 (inches per meter)

# 辺の長さを計算し、mmに変換
lengths = [distance(vertices[i], vertices[(i + 1) % 4]) / pixels_per_meter * 1000 for i in range(4)]


# 角度を計算
def angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


angles = [angle(vertices[i], vertices[(i + 1) % 4], vertices[(i + 2) % 4]) for i in range(4)]

# 画像に頂点と辺を描画
for i in range(4):
    cv2.circle(image, vertices[i], 30, (0, 0, 255), -1)
    cv2.line(image, vertices[i], vertices[(i + 1) % 4], (0, 0, 255), 10)

# 検出された点と線を含む画像を保存
output_path = r"images\detected_parallelogram.png"
cv2.imwrite(output_path, image)

# 結果を出力
print("頂点:", vertices)
print("辺の長さ:", lengths)
print("角度:", angles)
