# pytorch_cnn
Training Convolutional Neural Networks to Categorize LithiumBattery with PyTorch

## 圖像樣本
AOI檢測鋰電池X-Ray圖像後會有四類缺陷：上、下、NG、OK，如下圖(經廠商同意公開)
classes = ('bottom_NG', 'bottom_OK', 'top_NG', 'top_OK') = [0, 1, 2, 3]

class | labeled | 電池圖像 |
:----------: | :----------: | :----------: |
bottom_NG| 0 | <img alt="bottom_NG-0" src="https://github.com/dvsseed/pytorch_cnn/blob/master/bottom_NG_0.bmp" width="200" height="200">|
bottom_NG| 0 |<img alt="bottom_NG-1" src="https://github.com/dvsseed/pytorch_cnn/blob/master/bottom_NG_1.bmp" width="200" height="200">|
bottom_OK| 1 |<img alt="bottom_OK-0" src="https://github.com/dvsseed/pytorch_cnn/blob/master/bottom_OK_0.bmp" width="200" height="200">|
top_NG| 2 |<img alt="top_NG-0" src="https://github.com/dvsseed/pytorch_cnn/blob/master/top_NG_0.bmp" width="200" height="200">|
top_NG| 2 |<img alt="top_NG-1" src="https://github.com/dvsseed/pytorch_cnn/blob/master/top_NG_1.bmp" width="200" height="200">|
top_OK| 3 |<img alt="top_OK-0" src="https://github.com/dvsseed/pytorch_cnn/blob/master/top_OK_0.bmp" width="200" height="200">|

## 程式
網路架構 | 程式集 | 說明 |
:----------: | :----------: | :----------: |
CNN三層卷積 | cnn_l3_battery4.py + model1.py + tools1.py + dataset5.py | cnn_lx_battery: 主程式, model: 模型, tools: 工具集, dataset: 資料集+資料預處理 |
CNN四層卷積 | cnn_l4_battery4.py + model5.py + tools1.py + dataset5.py | 同上 |

## 技巧
* 本程式可自動判定 **GPU** 或 **CPU** 切換
* 使用 **tensorboardX** 及 **tensorboard_logger** 做視覺化記錄
* 使用 **adabound**(Based on Luo et al.[1]) 增加準確率
* 使用 **psutil** 記錄電腦資源使用狀態: **CPU**、**Memory**、**Disk I/O**等
* 當資料集比例(訓練:測試=80:20)時，Accuracy可達 100%, 其訓練100epochs之Accuracy及Loss function走勢圖, 如下
<img alt="training_accuracy" src="https://github.com/dvsseed/pytorch_cnn/blob/master/training_accuracy1.png" width="300" height="200">
<img alt="training_loss" src="https://github.com/dvsseed/pytorch_cnn/blob/master/training_loss1.png" width="300" height="200">

## References
[1] L. Luo, Y. Xiong, and Y. Liu, “Adaptive gradient methods with dynamic bound of learning rate,” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=Bkg3g2R9FX
