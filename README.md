# Invisible Gas Detection: An RGB-Thermal Cross Attention Network and A New Benchmark 
*Accepted to CVIU (JCR Q1, CCF rank B)* 🎉🎉🥳🥳
### [Paper](https://arxiv.org/abs/2403.17712) | [Gas-DB](https://drive.google.com/drive/folders/11t324MSRVQhptfLLu65MlPaSaPOJRf4Z)

## Authors

- [Jue Wang*](https://a-new-b.github.io/)<sup>1,3</sup>, [Yuxiang Lin*](https://lum1104.github.io/)<sup>2</sup>, Qi Zhao<sup>3</sup>, Dong Luo<sup>3</sup>, Shuaibao Chen<sup>3</sup>, Wei Chen<sup>3</sup>, [Xiaojiang Peng](https://pengxj.github.io/)<sup>2</sup>
- <sup>1</sup>[Southern University of Science and Technology](https://www.sustech.edu.cn/en/), <sup>2</sup>[Shenzhen Technology University](https://english.sztu.edu.cn/), <sup>3</sup>[Shenzhen Institute of Technology, CAS](https://www.siat.ac.cn/) [Jue and Yuxiang contribute equally to this work.]

If you are interested in our work, please star ⭐ our project.

## Training dataset preparation
- Prepare our Gas-DB dataset: please download in [Gas-DB](https://drive.google.com/file/d/1NkN1K41KmDhf3wuyi5W9UNDK1oV09xKb/view?usp=sharing).

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
@article{WANG2024104099,
  title = {Invisible gas detection: An RGB-thermal cross attention network and a new benchmark},
  journal = {Computer Vision and Image Understanding},
  pages = {104099},
  year = {2024},
  issn = {1077-3142},
  doi = {https://doi.org/10.1016/j.cviu.2024.104099},
  url = {https://www.sciencedirect.com/science/article/pii/S1077314224001802},
  author = {Jue Wang and Yuxiang Lin and Qi Zhao and Dong Luo and Shuaibao Chen and Wei Chen and Xiaojiang Peng},
}
```
