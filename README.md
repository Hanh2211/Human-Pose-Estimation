# Pose Estimation

### Description
Human Pose Estimation, sử dụng 2 model: pretrain yoloV11-pose và RCNN-Resnet top-down model (train từ đầu)

### Dataset

[COCO2017-Dataset](https://cocodataset.org/)
- [train2017](http://images.cocodataset.org/zips/train2017.zip)
- [val2017](http://images.cocodataset.org/zips/val2017.zip)
- [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [Data Format](https://cocodataset.org/#format-data)

#### Cấu trúc folder

```
root
├───coco
│   ├───annotations
│   │   ├───person_keypoints_train2017.json
│   │   └───person_keypoints_val2017.json
│   ├───train2017
│   │   └───(...).jpg
│   └───val2017
│       └───(...).jpg
├───weight.pth
...
```

### Kết quả

###### Google Drive:
- [weight.pth](https://drive.google.com/drive/folders/1VQ2k9rojHdJ3aLMdaGpcC_HP61UuBdT3?usp=sharing)

### Tham khảo

- https://pytorch.org/vision/main/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn
- https://cocodataset.org
