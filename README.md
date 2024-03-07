# EM-VLM4AD
* This repository contains the code necessary to replicate the paper "Efficient, Lightweight Multi-Frame Vision Language Model for Visual Question Answering in Autonomous Driving.
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
## Dataset
First download the train/val/test split [here](https://drive.google.com/file/d/1TyqlEY8_4lark86Y2cqUUMgCyCJvvFjN/view?usp=sharing) in your root folder. This will include data from the DriveLM dataset as well as the train/val/test splits we use for our experiments. The folder structure should be as follows:
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
```
