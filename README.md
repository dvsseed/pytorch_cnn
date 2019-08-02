# pytorch_cnn
Training Convolutional Neural Networks to Categorize LithiumBattery with PyTorch

AOI檢測鋰電池X-Ray圖像後會有四類缺陷：上、下、NG、OK，如下圖(經廠商同意公開)
classes = ('bottom_NG', 'bottom_OK', 'top_NG', 'top_OK') = [0, 1, 2, 3]

bottom_NG<img alt="bottom_NG-0" src="https://github.com/dvsseed/pytorch_cnn/blob/master/bottom_NG_0.bmp" width="50" height="50">, labeled: 0
bottom_NG<img alt="bottom_NG-1" src="https://github.com/dvsseed/pytorch_cnn/blob/master/bottom_NG_1.bmp" width="50" height="50">, labeled: 0
bottom_OK<img alt="bottom_OK-0" src="https://github.com/dvsseed/pytorch_cnn/blob/master/bottom_OK_0.bmp" width="50" height="50">, labeled: 1

top_NG<img alt="top_NG-0" src="https://github.com/dvsseed/pytorch_cnn/blob/master/top_NG_0.bmp" width="50" height="50">, labeled: 2
top_NG<img alt="top_NG-1" src="https://github.com/dvsseed/pytorch_cnn/blob/master/top_NG_1.bmp" width="50" height="50">, labeled: 2
top_OK<img alt="top_OK-0" src="https://github.com/dvsseed/pytorch_cnn/blob/master/top_OK_0.bmp" width="50" height="50">, labeled: 3

