# Project Title: From Microscope to AI: Developing an Integrated Diagnostic System for Endometrial Cytology
Immediate Object Detection of Endometrial Cytology under a Microscope using YOLOv5x

## Description
This repository hosts the implementation of an AI-assisted model designed to support the diagnosis of endometrial cytology through real-time object-detection technology. In the realm of microscopic cytological diagnosis, this study explores the feasibility of integrating AI technology directly into the existing diagnostic workflows without relying on whole slide imaging (WSI). 
This application of AI aims to demonstrate how it can be seamlessly incorporated to assist with real-time diagnostics under a microscope, which could be particularly beneficial in resource-limited settings and during time-sensitive procedures such as rapid on-site evaluation (ROSE). By maintaining compatibility with current workflows, this approach holds the potential to facilitate and enhance diagnostic processes across various medical fields, not limited to endometrial cytology.

## System, Software and Microscope Setup
- **Microscope**: ECLIPSE Ci (Nikon Co., Tokyo, Japan)
- **CCD Camera**: JCS-HR5U (CANON Inc., Tokyo, Japan)
- **Anaconda**: Distribution (version 2022.10)
- **Python**: 3.10.9
- **Deep Learning Framework**: PyTorch 1.13.1
- **YOLOv5**:Utilized for object detection, adapted for our specific needs in endometrial cytological analysis.
- **IDE**: Visual Studio Code (Microsoft Co., WA, USA)

While these specific devices and software were used in our study, the implemented AI model is compatible with a range of microscopes, CCD cameras, and development environments that can support PyTorch and YOLOv5x. Users are encouraged to adapt the setup based on the available equipment and software configurations to explore the potential application of this technology in various settings.


## Installation
First, install Anaconda (version 2022.10) from the official [Anaconda Website](https://www.anaconda.com/). Then, set up your environment using the following commands:

```bash
conda create -n yolov5x python=3.10.9
conda activate yolov5x
```

## Clone the YOLOv5 repository and install the required dependencies:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

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

## Quick Start: Using Our Trained Model for Inference
If you wish to use our trained model for immediate inference on your endometrial cytology images, you can download the trained weights and use the following steps:

1. **Download Our Trained Weights**
   - Download our best performing model weights from this [best.pt](https://github.com/ahp-aig/yolo_for_endometrial_cytology/raw/main/best.pt).

2. **Setup Your Environment**
   - Ensure your environment is set up according to the installation instructions provided earlier in this document.

3. **Run Inference**
   - Use the following command to run inference using the trained model in your Anaconda virtual environment within Visual Studio Code. This command will use the input from your microscope's CCD camera connected via USB, and display the real-time detection results with bounding boxes and confidence scores (CS) of 0.225 or higher on your monitor.
   ```bash
   python detect.py --source 0 --weights path_to_downloaded_weights/best.pt --conf 0.225

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
@article{yourname2024,
  title={From Microscope to AI: Developing an Integrated Diagnostic System for Endometrial Cytology},
author={Mika Terasaki, et al.},
journal={Journal Name},
year={2024}
}
```
## Privacy and Compliance Notice
The cytological images used in our study are not publicly available due to patient privacy concerns and hospital policy compliance. This section serves to remind users that when working with medical data, especially sensitive patient information, it is crucial to adhere to all applicable laws and regulations regarding data privacy and security.
