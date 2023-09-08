# ICU-Monitor-Vital-Extractor
This project focuses on extracting Vitals information, like heart rate, respiratory rate, blood pressures, from an image of an ICU monitor. The model uses logic based mapping of vitals thus making it robust to monitor-type changes. Also, the detection model is based on YOLOv7 which supports real-time detection, thus it can be used with surveillance cameras.      

# Screen Detection
YOLOv7: https://github.com/WongKinYiu/yolov7

hyperparameters: lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.3, cls_pw=1.0, obj=0.7, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.2, scale=0.9, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.0, paste_in=0.15, loss_ota=1

image_size = 416 Batch_size = 4 No. of epochs = 10

Final_precision: 0.93 Final_recall: 0.97 mAP@0.5: 0.995 mAP@0.5:0.95: 0.96

# Screen Classifier 
Model used: mobilenetV3_small

hyperparameters: lr=0.001, epochs=10, grad_clip=0.1, weight_decay=1e-4 Optimizer used: SGD

Final Validation Accuracy: 82%

# Heart Rate Graph Detection
YOLOv5: https://github.com/ultralytics/yolov5

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0

image_size: 416 Batch size: 16 No. of epochs: 10

Final_precision: 0.95 Final_recall: 0.97 mAP@0.5: 0.995
