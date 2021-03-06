Kaggle "Digit Recognizer" using Caffe (only GPU)
=========================================

This accuracy on Kaggle PL is about 0.99585

```
python convert_data.py -k 10
sh train_net.sh
python predict.py > result.csv
```
![Alt text](https://github.com/jasoncs-lin/Kaggle-DigitRecognizer/blob/master/top10.jpg)

References: 
1. <a href="https://github.com/slaypni/Kaggle-DigitRecognizer"> Caffe with LeNet on MNIST for Kaggle: slaypni/Kaggle-DigitRecognizer</a>
2. <a href="https://github.com/shicai/MobileNet-Caffe">MobileNet-Caffe: shicai/MobileNet-Caffe</a>
3. <a href="https://github.com/yonghenglh6/DepthwiseConvolution">Depthwise Convolutional Layer yonghenglh6/DepthwiseConvolution</a>
