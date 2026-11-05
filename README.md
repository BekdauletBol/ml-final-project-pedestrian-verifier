# Final Project — Machine Learning Verifier for YOLOv8n Pedestrian Detections

Student: **Bolatov Bekdaulet, SE-2405** (solo project) · Course: Machine Learning (AMO 3208) · Supervisor: Mimenbayeva A.B.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BekdauletBol/ml-final-project-pedestrian-verifier/blob/main/Final_Project_Bolatov_Bekdaulet.ipynb)

My diploma system [public.safe](https://github.com/BekdauletBol/public_final) uses phones as cameras, and a laptop
runs **YOLOv8n** to find people in danger zones. YOLOv8n sometimes calls a toy, a statue or an animal a "person",
and every such box inside a zone is a false alarm. In this project I compare **three machine learning models** that
look at the crop of every YOLOv8n box and say **pedestrian**, **other_object** or **background**. Only pedestrians go
to the zone rules.

The project joins my Assignment 2 (dataset), my endterm (CNN and MobileNetV2) and the new parts of the final
project: a classical model, a comparison of preprocessing methods, a comparison of the three models, and a web
application.

## Results (test set, 3,085 crops)

| Model | Accuracy (95 % CI) | Macro F1 | Pedestrian precision | Pedestrian recall | Model file |
|---|---|---|---|---|---|
| Model 1: HOG + PCA + RBF SVM | 0.784 (0.768–0.798) | 0.779 | 0.678 | 0.718 | 44.7 MB |
| Model 2: own CNN (improved) | 0.853 (0.841–0.865) | 0.850 | 0.822 | 0.810 | 1.0 MB |
| **Model 3: MobileNetV2 (fine-tuned)** | **0.928 (0.920–0.937)** | **0.928** | **0.925** | **0.941** | 9.2 MB |

Preprocessing for the classical model (linear SVM, test accuracy): raw pixels 0.484 → **HOG 0.745** → HOG + ANOVA
top-800 0.738 → HOG + PCA (469 features) 0.744. The three required CNN configurations reach 0.805 / 0.805 / 0.808.

**Practical test on real YOLOv8n boxes** (738 test photos, 1,184 boxes: 939 true, 245 false by the COCO labels):

| Filter | True pedestrians kept | False positives removed | Precision | Recall |
|---|---|---|---|---|
| YOLOv8n only | 939 (100 %) | 0 % | 0.793 | 0.603 |
| + HOG + SVM | 729 (77.6 %) | 32.2 % | 0.815 | 0.468 |
| + own CNN | 796 (84.8 %) | 20.4 % | 0.803 | 0.511 |
| **+ MobileNetV2** | **903 (96.2 %)** | 13.1 % | 0.809 | 0.580 |

A manual check showed that most "false positives" that MobileNetV2 keeps are real people without a COCO label
(27 of 40), so the real gain is larger than the measured one.

## Files

| Path | What it is |
|---|---|
| `Final_Project_Bolatov_Bekdaulet.ipynb` | the notebook with all steps and outputs (sections 3, 4, 5 = models 1, 2, 3) |
| `report/` | the report (PDF and Word) |
| `app.py` | **web application** (Gradio): upload a photo, choose a model, see kept and rejected boxes |
| `src/classical.py` | Model 1: raw-pixel and HOG features, feature selection, SVM / logistic regression / random forest |
| `src/models.py`, `src/train.py` | Model 2 (`PedCNN`) and Model 3 (`mobilenet_v2_transfer`), training loop |
| `src/data.py`, `src/evaluate.py` | loading, augmentation, normalization; metrics and plots |
| `src/verifiers.py`, `src/verified_detector.py` | the three models behind one interface; drop-in detector for public.safe |
| `scripts/make_yolo_detections.py` | runs YOLOv8n on the test photos and marks every box as true or false |
| `data/pedcrops_v1/` | **the dataset**: `train.zip`, `val.zip`, `test.zip` (JPEG crops by class), `metadata.csv`, `stats.json` |
| `data/yolo_test_detections/` | the 1,184 YOLOv8n boxes (crops, CSV) |
| `models/` | trained models: `hog_svm.joblib`, `pedcnn_*.pt`, `mobilenetv2_*.pt`, and `yolov8n.pt` |
| `results/` | result tables (CSV) and all figures |
| `app_examples/` | four COCO val2017 test photos for the web application |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Final_Project_Bolatov_Bekdaulet.ipynb   # TRAIN = False loads my CNN weights; the SVM is trained (a few minutes)
python app.py                                            # web application: http://127.0.0.1:7860
python app.py --share                                    # public link, for example from Google Colab
```

## Data and licenses

The dataset is made from COCO val2017 (Lin et al., 2014): annotations under CC BY 4.0, photos from Flickr under
their own licenses (see [cocodataset.org](https://cocodataset.org/#termsofuse)). YOLOv8n weights: Ultralytics (AGPL-3.0).
