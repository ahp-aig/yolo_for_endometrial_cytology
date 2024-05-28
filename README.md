# Project Title: From Microscope to AI: Developing an Integrated Diagnostic System for Endometrial Cytology

Real-Time Object Detection of Endometrial Cytology under a Microscope using YOLOv5x

## Description

This repository hosts the implementation of an AI-assisted model designed to support the diagnosis of endometrial cytology through real-time object-detection technology. In the realm of microscopic cytological diagnosis, this study explores the feasibility of integrating AI technology directly into the existing diagnostic workflows without relying on whole slide imaging (WSI). 

## System, Software and Microscope Setup

- **Microscope**: ECLIPSE Ci (Nikon Co., Tokyo, Japan)
- **CCD Camera**: JCS-HR5U (CANON Inc., Tokyo, Japan)
- **Anaconda**: Distribution (version 2022.10)
- **Python**: 3.10.9
- **Deep Learning Framework**: PyTorch 1.13.1
- **YOLOv5**:Utilized for object detection, adapted for our specific needs in endometrial cytological analysis.
- **IDE**: Visual Studio Code (Microsoft Co., WA, USA)

While these specific devices and software were used in our study, the implemented AI model is compatible with a range of microscopes, CCD cameras, and development environments that can support PyTorch and YOLOv5x. Users are encouraged to adapt the setup based on the available equipment and software configurations to explore the potential application of this technology in various settings.

## Quick Start: Using Our Trained Model for Inference

If you wish to use our trained model for real-time inference on your endometrial cytology images, follow these steps:

### 1. **Clone the Repository**

```bash
git clone https://github.com/ahp-aig/yolo_for_endometrial_cytology.git
cd yolo_for_endometrial_cytology
```

### 2. **Download Our Trained Weights**

Download our best performing model weights from this [best.pt](https://github.com/ahp-aig/yolo_for_endometrial_cytology/raw/main/best.pt) and place it in the cloned repository directory.

### 3. **Setup Your Environment**

Ensure you have Anaconda installed. If not, download and install it from the [Anaconda Website](https://www.anaconda.com/).

Create and activate a new Anaconda environment:

```bash
conda create -n yolov5x python=3.10.9
conda activate yolov5x
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 4. **Run Inference**

Use the following command to run inference using the trained model in your Anaconda virtual environment within Visual Studio Code. This command will use the input from your microscope's CCD camera connected via USB, and display the real-time detection results with bounding boxes and confidence scores of 0.225 or higher on your monitor. *Please adjust the confidence score setting to match your specific environment. 

```bash
python detect.py --source 0 --weights best.pt --conf 0.225
```

By running this command, the live feed from the CCD camera connected to the microscope will be displayed on the monitor. The trained model will detect abnormal cell clusters in real-time, showing bounding boxes with a confidence score of 0.225(or your confidence score setting) or higher.

## Training Your Own Model with Yolov5x for Real-Time Microscopic Detection

To train your own Yolov5x model for real-time microscopic object detection using your custom dataset, follow these detailed steps. This guide assumes you have already set up your development environment as described in previous sections.

### Step 1: Prepare Your Dataset

For effective training, your dataset should be properly annotated. Ensure each image in your dataset has corresponding labels with bounding boxes around the objects of interest, such as abnormal cell clusters. 

- **Annotation Format**: Use tools like [LabelImg](https://github.com/tzutalin/labelImg) to annotate your images in YOLO format, where each annotation is stored in a separate `.txt` file in the same directory as the image. Note that benign (normal) cells and other materials should be labeled as "background."
- **Dataset Structure**: Organize your dataset into three folders: `train`, `valid`, and `test`. Each folder should contain images and their corresponding annotation files.

## Clone the YOLOv5 repository and install the required dependencies

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### Step 2: Configure Your Training Environment

Before you start training, ensure your training environment is ready. Update the `data.yaml` file to point to your dataset directories and define the number of classes and class names.

Example of `data.yaml`:

```yaml
train: /path/to/your/dataset/train
val: /path/to/your/dataset/valid
test: /path/to/your/dataset/test

# Classes
nc: 1  # Number of classes (e.g., only 'malignant' as 'benign' is background)
names: ['malignant']
```

nc: This parameter (nc) represents the number of classes. In this case, it is set to 1 because only the 'malignant' class is used for detection, while 'benign' is treated as background.
names: This list under names specifies the class labels that your model will predict. Ensure that this list matches the labels used during the annotation of your dataset.

## Acquisition of Digital Images

- **Device Used**: iPhone SE (Apple Inc., Cupertino, CA, USA) mounted on an Olympus BX53 microscope (EVIDENT/Olympus, Tokyo, Japan).
- **Adapter**: i-NTER LENS (MICRONET Co., Kawaguchi, Saitama, Japan).
- **Resolution**: Images captured at 4,032 × 3,024 pixels.
- **Settings**: Manual focus adjustment with the objective lens set to 20x magnification.
- **Image Selection**: For malignant cases, images centered on abnormal cell clusters identified by a gynecologic pathologist; for benign cases, random selections of all visible cells.

## Dataset Preparation Guidelines

Prepare your dataset by categorizing images into benign and malignant types. For annotations, use tools like [LabelImg](https://github.com/HumanSignal/labelImg/tree/master), a graphical image annotation tool, to mark areas of interest such as abnormal cell clusters. Ensure that unannotated regions are labeled as "background." We recommend dividing your data into training, validation, and testing sets using an 8:1:1 ratio. This setup will help mimic the conditions under which our model was developed and tested.

## Training the Model

After preparing your dataset, you can train the model using the following command:

```bash
python train.py --batch-size 4 --epochs 200 --data path_to_your_data\data.yaml --weights best_model_weights.pt
```

Replace 'path_to_your_data/data.yaml' with the path to your dataset configuration file and 'path_to_initial_weights/weights_file.pt' with the path to your initial weights file, if available. This command initializes the training process using your own data, allowing you to adapt and optimize the model according to your specific needs.

### Hyperparameters and Training Details

We have adapted the training process according to specific needs for cytological image analysis. Below are the hyperparameters used:

- **Learning Rate (lr0)**: 0.01
- **Final Learning Rate (lrf)**: 0.1
- **Momentum**: 0.937
- **Weight Decay**: 0.0005
- **Warmup Epochs**: 3.0
- **Batch Size**: 4
- **Epochs**: 200

Full details on the hyperparameters can be found in `hyp.scratch-high.yaml`.

## Additional Resources

For more detailed information on modifying and using YOLOv5 for different scenarios, please refer to the [official YOLOv5 GitHub repository](https://github.com/ultralytics/yolov5).

## License

This project is licensed under the GPL-3.0 license. Please see the [LICENSE](LICENSE) file for more details.

## Citation

If you use this project in your research, please cite it as follows:

```bibtex
@article{Mika Terasaki2024,
  title={From Microscope to AI: Developing an Integrated Diagnostic System for Endometrial Cytology},
author={Mika Terasaki, Shun Tanaka, Ichito Shimokawa et al.},
journal={Journal Name},
year={2024}
}
```

## Privacy and Compliance Notice

The cytological images used in our study are not publicly available due to patient privacy concerns and hospital policy compliance. Please note that the use of AI tools and software in clinical diagnostics must comply with local medical device regulations and laws. The pre-trained model provided here is intended for research and development purposes only and should not be used as a diagnostic tool without proper validation and regulatory approval. Users are responsible for ensuring that the use of this model complies with all applicable laws and standards in their jurisdiction. Any clinical application or diagnostic use of the model requires further validation and must adhere to the necessary regulatory approvals.
