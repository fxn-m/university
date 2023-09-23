# CellDetect ♻

💡 **Idea**: Detect lithium ion cells in images of modules, enabling fully automated recycling of retired batteries. The centre of each cell face can be milled, and cells extracted for re-purposing!

🛠️ **Approach**: Fine tune <a href="https://github.com/ultralytics/yolov5">YOLOv5</a> on a hand-labelled dataset ( > 1,000 labels! ) to detect cells.

✅ **Result**: 1.00 mAP_0.5 & 0.606 mAP_0.5:0.95 detecting cells on batteries not included in training data set.


📄 **Report**: The project report can be found <a href="https://drive.google.com/file/d/1jc1pwSu0KWL3z2xB4xRNDjy2_dmBP5Kq/view?pli=1">here.</a>
