from ultralytics import YOLO

model = YOLO('yolo11x.pt')  

results = model.train(data = 'FruitDetective.yaml', epochs =8, batch =8)
