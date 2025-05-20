import cv2
from ultralytics import YOLO

# 1. 加载模型（替换为你的模型路径）
model = YOLO(r"D:\daima\python\mask\runs\train\exp9\weights\best.pt")

# 2. 打开摄像头（默认使用 0，外接摄像头可能是 1 或 2）
cap = cv2.VideoCapture(0)

# 3. 检查摄像头是否打开成功
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 4. 实时处理
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧")
        break

    # 使用模型进行推理
    results = model(frame, verbose=False)

    # 提取检测框
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        label = results[0].names[cls_id]

        # 绘制矩形和标签
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow("YOLOv8 实时检测", frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
