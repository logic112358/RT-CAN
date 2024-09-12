# Invisible Gas Detection: An RGB-Thermal Cross Attention Network and A New Benchmark 
*Accepted to CVIU (JCR Q1, CCF rank B)* 🎉🎉🥳🥳
### [Paper (Latest version)](https://www.sciencedirect.com/science/article/abs/pii/S1077314224001802) | [Gas-DB](https://drive.google.com/drive/folders/11t324MSRVQhptfLLu65MlPaSaPOJRf4Z)

## Authors

- [Jue Wang*](https://a-new-b.github.io/)<sup>1,3</sup>, [Yuxiang Lin*](https://lum1104.github.io/)<sup>2</sup>, Qi Zhao<sup>3</sup>, Dong Luo<sup>3</sup>, Shuaibao Chen<sup>3</sup>, Wei Chen<sup>3</sup>, [Xiaojiang Peng](https://pengxj.github.io/)<sup>2</sup>
- <sup>1</sup>[Southern University of Science and Technology](https://www.sustech.edu.cn/en/), <sup>2</sup>[Shenzhen Technology University](https://english.sztu.edu.cn/), <sup>3</sup>[Shenzhen Institute of Technology, CAS](https://www.siat.ac.cn/) [Jue and Yuxiang contribute equally to this work.]

If you are interested in our work, please star ⭐ our project.

## Training dataset preparation
- Prepare our Gas-DB dataset: please download in [Gas-DB](https://drive.google.com/file/d/1NkN1K41KmDhf3wuyi5W9UNDK1oV09xKb/view?usp=sharing).
## Code
### Setup
```
conda create -n RT-CAN python==3.8.16
conda activate RT-CAN
pip install -r requirements.txt
```
### Train
```
python train.py
```
### Test
```
python test.py
```
### Infetence
```
python inference.py
```
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
title = {Invisible gas detection: An RGB-thermal cross attention network and a new benchmark},
journal = {Computer Vision and Image Understanding},
volume = {248},
pages = {104099},
year = {2024},
issn = {1077-3142},
doi = {https://doi.org/10.1016/j.cviu.2024.104099},
url = {https://www.sciencedirect.com/science/article/pii/S1077314224001802},
author = {Jue Wang and Yuxiang Lin and Qi Zhao and Dong Luo and Shuaibao Chen and Wei Chen and Xiaojiang Peng},
keywords = {Gas detection, Computer vision, RGB-Thermal, Gas-DB},
abstract = {The widespread use of various chemical gases in industrial processes necessitates effective measures to prevent their leakage during transportation and storage, given their high toxicity. Thermal infrared-based computer vision detection techniques provide a straightforward approach to identify gas leakage areas. However, the development of high-quality algorithms has been challenging due to the low texture in thermal images and the lack of open-source datasets. In this paper, we present the RGB-Thermal Cross Attention Network (RT-CAN), which employs an RGB-assisted two-stream network architecture to integrate texture information from RGB images and gas area information from thermal images. Additionally, to facilitate the research of invisible gas detection, we introduce Gas-DB, an extensive open-source gas detection database including about 1.3K well-annotated RGB-thermal images with eight variant collection scenes. Experimental results demonstrate that our method successfully leverages the advantages of both modalities, achieving state-of-the-art (SOTA) performance among RGB-thermal methods, surpassing single-stream SOTA models in terms of accuracy, Intersection of Union (IoU), and F2 metrics by 4.86%, 5.65%, and 4.88%, respectively. The code and data can be found at https://github.com/logic112358/RT-CAN.}
}
```
