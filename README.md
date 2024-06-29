# Invisible Gas Detection: An RGB-Thermal Cross Attention Network and A New Benchmark

[Paper](https://arxiv.org/abs/2403.17712) | [Gas-DB](https://drive.google.com/drive/folders/11t324MSRVQhptfLLu65MlPaSaPOJRf4Z)

If you are interested in our work, please star ⭐ our project.

## Training dataset preparation
- Prepare our Gas-DB dataset: please download in [Gas-DB](https://drive.google.com/drive/folders/11t324MSRVQhptfLLu65MlPaSaPOJRf4Z).

Code will be made available soon. Stay tuned!

## RGB-Thermal Cross Attention Network
![ Illustration the architecture of RGB-Thermal Two Stream Cross Attention Network. (a) Two stream RGB-ThermaR Cl Encoder, (b) Cascaded Decoder.](https://github.com/logic112358/image/blob/main/20240529163048.png)

Illustration the architecture of RGB-Thermal Two Stream Cross Attention Network. (a) Two stream RGB-ThermaR Cl Encoder, (b) Cascaded Decoder.
## An overview of our Gas-DB  
![an overview of our Gas-DB](https://github.com/logic112358/image/blob/main/20240529155658.png)

This figure shows an overview of our Gas-DB, containing 8 kinds of scenery, containing sunny, rainy, double leakage, nearly leakage, further leakage, overlook, simple background, and complex background. The last one is the original gas image without manually annotating.

##  The comparision of the GasVid and our Gas-DB.
![The comparision of the GasVid and our Gas-DB](https://github.com/logic112358/image/blob/main/20240529155716.png)

## The visualization of the prediction comparisons from different methods
![The visualization of the prediction comparisons from different methods](https://github.com/logic112358/image/blob/main/20240529155745.png)

The visualization of the prediction comparisons from different methods, according to the rows from top to bottom in order: RGB; Thermal; Ground Truth; PSPNet; Segformer; YOLOv5; MFNet; EAEFNet; Ours.

## Contact   
For any question, feel free to email <j.wang2@siat.ac.cn> and <yuxiang.lin@gatech.edu>.

## Citation
```
@article{RT-CAN,
  title={Invisible Gas Detection: An RGB-Thermal Cross Attention Network and A New Benchmark},
  author={Wang, Jue and Lin, Yuxiang and Zhao, Qi and Luo, Dong and Chen, Shuaibao and Chen, Wei and Peng, Xiaojiang},
  journal={arXiv preprint arXiv:2403.17712},
  year={2024}
}
```
