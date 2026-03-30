# Dataset Documentation

## Dataset Overview

This project uses a custom dataset designed to train a computer vision system for monitoring construction site safety. The dataset focuses on detecting workers and identifying whether they are wearing required personal protective equipment (PPE), specifically helmets and high-visibility vests.

The dataset combines publicly available construction safety images with additional images collected and curated manually to extend the dataset.

---

## Data Sources

The dataset was created using a combination of:

1. Public construction safety datasets from Kaggle and similar repositories.
2. Additional images collected from publicly available construction site photos and videos.
3. Video frames extracted from construction site footage.

These additional images were added to extend the dataset and create more varied examples of both safe and unsafe situations.

---

## Dataset Size

Initial labeled dataset:

* **Total labeled images:** 516

After dataset augmentation using Roboflow:

* **Total training dataset size:** approximately 1200 images

Dataset split used during training:

* **Training set:** ~1100 images
* **Validation set:** ~100 images
* **Test set:** ~50 images

---

## Classes

The dataset includes three object detection classes:

1. **person**
   Represents workers present in the construction scene.

2. **helmet**
   Represents safety helmets (hard hats) worn by workers.

3. **vest**
   Represents high-visibility safety vests worn by workers.

---

## Annotation Process

All images were annotated using **Roboflow**.

The annotation process included:

1. Drawing bounding boxes around each visible worker.
2. Labeling helmets worn by workers.
3. Labeling safety vests worn by workers.

Bounding box annotations follow the **YOLO object detection format**.

---

## Data Diversity

The dataset was intentionally curated to include variation across several dimensions.

### Environment

Images include multiple construction environments such as:

* indoor construction areas
* outdoor construction sites
* scaffolding areas
* open construction zones

### Lighting Conditions

Examples include:

* daylight
* shadowed environments
* indoor artificial lighting
* overcast outdoor conditions

### Safe and Unsafe Situations

The dataset intentionally contains a mixture of:

* **safe scenes** where workers wear required PPE
* **unsafe scenes** where helmets or vests are missing

This balance helps the model learn to distinguish between compliant and non-compliant situations.

---

## Data Augmentation

Roboflow preprocessing and augmentation were used to improve model generalization.

Applied transformations include:

* horizontal flipping
* brightness adjustment
* resizing images to 640 × 640 resolution

Augmentation helps the model learn to detect PPE under varying conditions.

---

## Limitations

Several limitations exist in the dataset:

* Some workers appear partially occluded in crowded scenes.
* Vest detection can be difficult when clothing colors resemble safety vests.
* Small or distant workers may be harder to detect.

These limitations represent potential areas for future dataset expansion.

---

