import cv2
import numpy as np
from PIL import Image

# DPI設定
dpi = 600
mm_to_inch = 1 / 25.4

# A3サイズの画像を作成 (A3: 297mm x 420mm)
a3_width_mm, a3_height_mm = 297, 420
a3_width = int(a3_width_mm * mm_to_inch * dpi)
a3_height = int(a3_height_mm * mm_to_inch * dpi)
image = np.zeros((a3_height, a3_width, 3), dtype=np.uint8)

# A4サイズの平行四辺形の頂点を計算 (A4: 210mm x 297mm)
a4_width_mm, a4_height_mm = 210, 297
a4_width = int(a4_width_mm * mm_to_inch * dpi)
a4_height = int(a4_height_mm * mm_to_inch * dpi)

offset_x, offset_y = (a3_width - a4_width) // 2, (a3_height - a4_height) // 2  # A3の中央に配置するためのオフセット

# 平行四辺形の頂点 (対角が89°と91°)
pts = np.array(
    [
        [offset_x, offset_y],
        [offset_x + a4_width, offset_y + int(a4_width * np.tan(np.radians(2)))],
        [offset_x + a4_width, offset_y + a4_height + int(a4_width * np.tan(np.radians(2)))],
        [offset_x, offset_y + a4_height],
    ],
    dtype=np.int32,
)

# 回転行列を計算 (1°時計回り)
center = (a3_width // 2, a3_height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, -3, 1.0)

# 頂点を回転
pts = np.hstack((pts, np.ones((4, 1), dtype=np.int32)))  # 同次座標系に変換
rotated_pts = rotation_matrix.dot(pts.T).T.astype(np.int32)

# 平行四辺形を白色で描画
cv2.fillPoly(image, [rotated_pts], (255, 255, 255))

# 画像をぼかす
blurred_image = cv2.GaussianBlur(image, (21, 21), 0)

# OpenCVの画像をPillowの画像に変換
image_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))

# 画像を保存 (DPI情報を追加)
image_pil.save(r"images\a3_with_a4_parallelogram_blurred_600dpi.png", dpi=(dpi, dpi))
