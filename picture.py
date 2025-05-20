# import cv2
# from ultralytics import YOLO
#
# # 1. 加载模型
# model = YOLO(r"D:\daima\python\mask\runs\train\exp5\weights\best.pt")
#
# # 2. 读取并调整图像
# image_path = r'D:\daima\python\mask\FaceMask.v3i.yolov11\test\images\b21f2584-fe04-4cb9-a499-a233c940a0a2-33c3c4d76d1f2ed41f18302ddfe103d1_jpeg.rf.87cbe0628b743fa9f17ffd7ed85501f8.jpg'
# image = cv2.imread(image_path)
#
# # 3. 推理与结果处理
# results = model(image)
# boxes = results[0].boxes
# labels = results[0].names
#
# # 4. 遍历检测结果并打印
# for box in boxes:
#     x1, y1, x2, y2 = map(int, box.xyxy[0])
#     confidence = box.conf[0].item()
#     label = labels[int(box.cls[0].item())]
#
#     # 打印每个检测结果
#     print(f"目标: {label}, 置信度: {confidence:.2f}")
#
#     # 绘制检测框
#     cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
# # 5. 恢复尺寸并显示
#
# cv2.imshow("Processed Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 6. 保存结果（可选）
# cv2.imwrite("processed_image.jpg", image)


import cv2
from ultralytics import YOLO

# 1. 加载模型
model = YOLO(r"D:\daima\python\mask\runs\train\exp9\weights\best.pt")

# 2. 读取图像
image_path = r'D:\daima\python\mask\Real-time Face Mask Detection and Validation System Dataset.v4i.yolov11\test\images\01061_Mask_Mouth_Chin_jpg.rf.14c02be08e63dba76f4e186b64b2c825.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"图片未读取成功，请检查路径：{image_path}")

# 3. 推理
results = model(image)
boxes = results[0].boxes
labels = model.names  # 使用 model.names 获取标签名

# 4. 处理结果
if boxes is None or len(boxes) == 0:
    print("⚠️ 没有检测到目标！")
else:
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confidence = float(box.conf[0])
        label = labels[int(box.cls[0])]

        print(f"目标: {label}, 置信度: {confidence:.2f}")

        # 绘制检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 5. 显示
cv2.imshow("Processed Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. 保存结果
cv2.imwrite("processed_image.jpg", image)
