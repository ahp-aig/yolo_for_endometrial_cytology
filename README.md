## Project Title: 
# From Microscope to AI: Developing an Integrated Diagnostic System with Real-Time Object Detection for Endometrial Cytology

## Description

This repository contains the implementation of an AI-assisted model for the diagnosis of endometrial cytology through real-time object detection. This study demonstrates the integration of AI technology into existing diagnostic workflows without the need for whole slide imaging (WSI).

## Important Note

Before using this repository, please adjust the settings, hyperparameters, and device choices according to your unique dataset and operating environment. This project was developed with specific hardware and dataset conditions; hence, optimal performance may require modifications tailored to your specific circumstances.

## Acquisition of Digital Images

- **Device Used**: iPhone SE (Apple Inc., Cupertino, CA, USA) mounted on an Olympus BX53 microscope (EVIDENT/Olympus, Tokyo, Japan).
- **Adapter**: i-NTER LENS (MICRONET Co., Kawaguchi, Saitama, Japan).
- **Resolution**: Images captured at 4,032 × 3,024 pixels.
- **Settings**: Manual focus, 20x magnification.
- **Image Selection**: Abnormal cell clusters for malignant cases; random cells for benign cases.

## Prepare Your Dataset

Proper annotation is crucial for effective training. Ensure each image is labeled with bounding boxes around objects of interest, such as abnormal cell clusters.

- **Annotation Format**: Use tools like [LabelImg](https://github.com/tzutalin/labelImg) for YOLO format annotations, stored as `.txt` files in the same directory as the image.
- **Dataset Structure**: Organize into `train`, `valid`, and `test` folders, each containing images and their annotations, and the images and labels should be put like below.
  - train  
    -images    
    -labels  
  - valid  
    -images    
    -labels   
  - test  
    -images  
  

## System, Software, and Microscope Setup

- **Microscope**: ECLIPSE Ci (Nikon Co., Tokyo, Japan)
- **CCD Camera**: JCS-HR5U (CANON Inc., Tokyo, Japan)
- **Anaconda**: Version 2022.10
- **Python**: 3.10.9
- **Deep Learning Framework**: PyTorch 1.13.1
- **YOLOv5**: Custom adapted for endometrial cytology.
- **IDE**: Visual Studio Code (Microsoft Co., WA, USA)

### Setup Your Environment

Download and install Anaconda from the [Anaconda Website](https://www.anaconda.com/):

```bash
conda create -n myenv python=3.10.9
conda activate myenv
```

## Clone the YOLOv5 repository and install the required dependencies

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### Configure Your Training Environment

Customize your training environment by updating the data.yaml file:

```yaml
train: /path/to/your/dataset/train  # Customize to your dataset location
val: /path/to/your/dataset/valid    # Customize to your dataset location
test: /path/to/your/dataset/test    # Customize to your dataset location

# Classes
nc: 1  # Customize number of classes if different
names: ['malignant']  # Customize class names based on your dataset labels
```

nc: This parameter (nc) represents the number of classes. In this case, it is set to 1 because only the 'malignant' class is used for detection, while 'benign' is treated as background in our study.
names: This list under names specifies the class labels that your model will predict. Ensure that this list matches the labels used during the annotation of your dataset.

## Training the Model

Train your model using:

```bash
python train.py --batch-size 4 --epochs 200 --data path_to_your_data\data.yaml --weights path_to_your_data/yolov5x/v5l/v5m/v5s.pt
```

Replace 'path_to_your_data/data.yaml' with the path to your dataset configuration file and 'path_to_initial_weights/weights_file.pt' with the path to your initial weights file, if available. This command initializes the training process using your own data, allowing you to adapt and optimize the model according to your specific needs. For other commands, refer to the official yolov5 documentation.

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

## Evaluate the Model

After training, evaluate the model's performance using the validation dataset. Use metrics such as precision, recall, and F1-score to assess the model's accuracy. For detailed evaluation methods, refer to our evaluation guide ((https://www.researchsquare.com/article/rs-4205271/v1)).
```bash
python val.py --data path_to_your_data\data.yaml --weights best_model_weights.pt 
```

### **Run Inference**

Use the following command to run inference using the trained model in your Anaconda virtual environment within Visual Studio Code. This command will use the input from your microscope's CCD camera connected via USB, and display the real-time detection results with bounding boxes and appropriate confidence scores you set or higher on your monitor. 

```bash
python detect.py --source 0 --weights best.pt --conf 0.5 # Adjust the confidence score as needed

```

## Additional Resources

Refer to [the official YOLOv5 GitHub repository[(https://github.com/ultralytics/yolov5) for more information.

## License

This project is licensed under the GPL-3.0 license. Please see the [LICENSE](LICENSE) file for more details.

## Citation

If you use this project in your research, please cite it as follows:

```bibtex
@article{Mika Terasaki2024,
  title={From Microscope to AI: Developing an Integrated Diagnostic System with Real-Time Object Detection for Endometrial Cytology},
author={Mika Terasaki, Shun Tanaka, Ichito Shimokawa et al.},
doi={https://doi.org/10.21203/rs.3.rs-4205271/v1},
year={2024}
}
```

## Privacy and Compliance Notice

The cytological images used in our study are not publicly available due to patient privacy concerns and hospital policy compliance. 

## Legal and Ethical Considerations

This AI model is provided for research purposes only and has not been approved for clinical use. Please ensure compliance with all local regulations and ethical standards before using this model in a clinical setting.
