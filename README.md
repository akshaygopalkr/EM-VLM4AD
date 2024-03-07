# EM-VLM4AD
<div style="display: flex;">
    <img src="assets/ex1.jpeg" alt="Image 1" style="width: 49%;">
    <img src="assets/ex2.jpeg" alt="Image 2" style="width: 49%;">
</div>

* This repository contains the code necessary to replicate the paper "Efficient, Lightweight Multi-Frame Vision Language Model for Visual Question Answering in Autonomous Driving", which was submitted to the Vision & Language for Autonomous Driving & Robotics Workshop at CVPR 2024.
## Installation
1. Clone this repository
2. In the repository directory, run `multi_frame_results`
3. Install the following libraries (assuming pytorch is properly installed in environment):
```
pip install peft
pip install transformers
pip install accelerate
pip install bitsandbytes
pip install pycocotools
pip install pycocoevalcap
```
## Model Weights
* You can download the model weights for the [T5-Medium](https://drive.google.com/drive/folders/1K61Ou-m5c5UmN2ggT-Huw3rv7PhW5Wft?usp=sharing) and [T5-Large-Q](https://drive.google.com/drive/folders/1bzxaxz6zSRZuMv284cjhQTSs_8i98kGI?usp=sharing) version of EM-VLM4AD at the following links. Put the folders for each of these models into the `multi_frame_results` folder. Your directory should look like the following:
```
└── rootFolder
 ├── multi_frame_results/
      ├── T5-Medium/
        ├── latest_model.pth
      ├── T5-Large/
        ├── latest_model.pth
```
## Dataset
First download the train/val/test split [here](https://drive.google.com/file/d/1TyqlEY8_4lark86Y2cqUUMgCyCJvvFjN/view?usp=sharing) in your root folder. This will include data from the DriveLM dataset as well as the train/val/test splits we use for our experiments. The folder structure should now be as follows: 
```
└── rootFolder
  ├── data/
    ├── multi_frame/
      ├── multi_frame_train.json
      ├── multi_frame_val.json
      ├── multi_frame_test.json
      ├── multi_frame_test_coco.json
      ├── image_id.json
    ├── QA_dataset_nus/
      ├── v1_0_train_nus.json
    ├── nuscenes/
      ├── samples/
  ├── multi_frame_results/
      ├── T5-Medium/
      ├── T5-Large/
```
## Running on Google Colab
If you want to run our code on Google Colab, we have provided three different notebooks 
