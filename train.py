import warnings
warnings.filterwarnings('ignore')



from ultralytics import YOLO

if __name__ == '__main__':
    # 使用预训练权重初始化（关键修改）
    model = YOLO(r'D:\daima\python\mask\yolo11n.pt')  # 使用官方预训练模型

    # 迁移学习训练配置
    results = model.train(
        data=r"D:\daima\python\mask\Real-time Face Mask Detection and Validation System Dataset.v4i.yolov11\data.yaml",
        epochs=10,  # 调整到合理周期
        imgsz=640,
        batch=16,
        patience=50 ,  # 添加早停机制
        device='0',
        optimizer='auto',  # 使用自动优化器选择
        lr0=0.01,  # 明确设置初始学习率
        amp=True,
        project='runs/train',
        name='exp',
        single_cls=False,  # 确认是否需要单类别
        pretrained=True, # 确保使用预训练权重
        workers = 0
    )