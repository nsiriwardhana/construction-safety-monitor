# Construction Safety Monitoring using YOLOv8

## Project Overview

This project implements an AI-based construction site safety monitoring system. The system detects workers and their personal protective equipment (PPE), including helmets and safety vests, using a YOLOv8 object detection model. After detection, a rule-based evaluation module determines whether each worker complies with safety requirements.

The system outputs both worker-level compliance and an overall scene-level safety status.

---

## Problem Statement

Construction sites require strict safety compliance. Workers must wear appropriate PPE such as helmets and high-visibility vests. Manual monitoring is inefficient and error-prone.

This project demonstrates how computer vision can automatically detect workers and PPE compliance using a trained deep learning model.

---

## Dataset

A custom dataset was created for this project.

Steps followed:

1. Images collected from public construction safety datasets and custom sources.
2. Dataset manually annotated with three classes:

   * person
   * helmet
   * vest
3. Annotation performed using Roboflow.
4. Dataset augmentation applied during dataset generation.
5. Final dataset split:

   * Train
   * Validation
   * Test

Total manually labeled images: **516**

After augmentation the training dataset contained approximately **1200 images**.

---

## Model Training

The object detection model was trained using **YOLOv8** from the Ultralytics framework.

Training configuration:

* Model: YOLOv8 Nano
* Epochs: 50
* Image size: 640
* Dataset format: YOLOv8

---

## Model Performance

Evaluation metrics from the test set:

* Precision: ~0.66
* Recall: ~0.66
* mAP50: ~0.60
* mAP50–95: ~0.38

Helmet detection achieved the highest accuracy, while vest detection showed slightly lower performance due to variation in clothing and lighting conditions.

---

## System Architecture

The system contains three main components:

### 1. Detection Layer

YOLOv8 model detects:

* person
* helmet
* vest

### 2. Rule Engine

Determines safety compliance for each worker.

Rules used:

* Worker is **SAFE** if helmet AND vest are detected.
* Worker is **UNSAFE** if any required PPE is missing.

### 3. Scene-Level Evaluation

The system aggregates worker results to determine whether the entire scene is safe or unsafe.

---

## Example Output

Worker-level results:

```
Worker 1: SAFE
Worker 2: UNSAFE (missing vest)
Worker 3: UNSAFE (missing helmet)
```

Scene-level summary:

```
Workers detected: 3
Safe workers: 1
Unsafe workers: 2
Scene status: UNSAFE
```

---

## Project Structure

```
construction-safety-monitor/
│
├── data/
│   └── sample_outputs/
│
├── models/
│   └── best.pt
│
├── notebooks/
│   └── construction_safety_monitor_training.ipynb
│
├── src/
│   ├── rule_engine.py
│   └── inference.py
│
├── docs/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run Inference

Install dependencies:

```
pip install -r requirements.txt
```

Run the safety analysis:

```
python src/inference.py
```

Then provide the image path when prompted.

---

## Future Improvements

Possible improvements include:

* Training with a larger dataset
* Improving vest detection accuracy
* Real-time monitoring using video streams
* Integration with construction site monitoring systems

---

## Technologies Used

* Python
* YOLOv8 (Ultralytics)
* Roboflow
* Google Colab
* OpenCV
