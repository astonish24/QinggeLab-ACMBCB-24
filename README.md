# QinggeLab-ACMBCB-24
Federated Learning and Knowledge distillation for Covid-19 detection from CT scans

Federated Learning Multi Teachers Knowledge Distillation (FL-MTKD) is a technique that enhances federated learning by incorporating knowledge from multiple teacher models. In this approach, decentralized devices train local models while keeping data private. These models, called teacher models, each learn from different data subsets. The student model then distills knowledge from these multiple teachers, leading to better generalization and more efficient learning. FL-MTKD combines the benefits of improved model performance and enhanced data privacy.

## Data Source

Hopital 1 (Datasets 1), Hospital 2 (Dataset2 2) and Hospital 3 (Datasets 3)

- https://github.com/ieee8023/covid-chestxray-dataset/tree/master/images (dataset3)
- https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/ (dataset3)

- https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset (dataset2)

- https://www.kaggle.com/datasets/mehradaria/covid19-lung-ct-scans (Dataset1)
- https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset (Dataset1)

Main files:
 - FL_MTKD_TeacherModelsTraining.ipynb: This file contains the training for the teacher models (CovidCNN, DeepCovid and CovidVGG16) on datasets. They are utilized as pretrained teacher models to distill knowledge to student model in the FL-MTKD algorithms.
 - FL_MTKD.ipynb: This file contain the implementation of the FL-MTKD algorithm and utillization of the teacher modesl to train student model (SimpCNN) in federated learning.
 - FL_SimpCNN.ipynb: In the file we train the SimpCNN using fedAvg for comparison with FL-MTKD.
 - FL_DeepCovid.ipynb: In the file we train the DeepCovid using fedAvg for comparison with FL-MTKD.
 - FL_CovidVGG16.ipynb: In the file we train the CovidVGG16 using fedAvg for comparison with FL-MTKD.
 - FL_CovidCNN.ipynb: In the file we train the CovidCNN using fedAvg for comparison with FL-MTKD.

Dependencies: python 3.9 tensorflow < 2.11 MS Visual Studio 2019 CUDA v.1.1X cuDNN v.8.1 miniconda

https://www.tensorflow.org/install/pip

Installing tensorflow<2.11 with conda, which is the last version of tensorflow that can run on Windows Native
As such, requires older versions of MS Visual Studio (2019), CUDA Toolkit 11.2, and cuDNN SDK 8.1.0
conda install: Install miniconda from the internet

conda create --name tf python=3.9
conda activate tf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
pip install "tensorflow<2.11" 
pip install tensorflow_federated
