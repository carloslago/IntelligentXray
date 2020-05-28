# IntelligentXray

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Packages](#packages)



<!-- ABOUT THE PROJECT -->
## About The Project

Final Degree Project for Computer Engineering, the project is oriented towards the diagnosis
of multiple chest pathologies with deep learning, studying the effectiveness of transfer
learning techniques. The datasets for the model development are acquired both by Kaggle
and CheXpert, a dataset provided by Stanford ML Group, with over 200.000 samples with 
both frontal and lateral X-rays and 14 different observations. 

First, the data is cleaned and prepared for the training process, where several architectures and 
techniques are tested, like transfer learning from deep convolutional neural networks
, such as DenseNet, image augmentation techniques and parallel CNNs. Then, an explanation
of the model is performed to detect how it decides whether a certain pathology exists or
not and what are the areas of interest for each pathology, comparing its thinking process
with state of the art methods used by doctors to check if the predictions have sense.

Finally, the obtained model is deployed on a web server, accesible in this [repo](https://github.com/carloslago/IntelligentXray_Server), which can be used to upload
an X-Ray to get a fast real-time analysis. An additional case motivated by
the current situation has also been studied, using transfer learning to detect 
COVID-19 and differentiate it from usual pneumonia, reaching an accuracy higher
than 90% and an AUC of 0.98.

### Built With
* [Python](https://www.python.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Lime](https://github.com/marcotcr/lime)

<!-- GETTING STARTED -->
## Getting Started

In order to train the models its neccesary to acquire the datasets being used and to install all the requirements.

### Datasets
The datasets used for the project are the following:
* [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
* [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
* [Pulmonary Chest X-Ray Abnormalities](https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities)
* [COVID-19 image dataset](https://github.com/ieee8023/covid-chestxray-dataset)

### Installation

1. Clone the repository
```sh
git clone https://github.com/carloslago/Ulcers_Prototype.git
```

2. Install Python packages
```sh
pip install requirements.txt
```



## Packages
* CheXpert: code for training models for the CheXpert challenge.
* Covid: COVID-19 network.
* Pneuomonia: network for the pneuomina kaggle challenge.
* XretAbnormalities: network with transfer learning from the pneumonia case.
* utils: scripts to reorder datasets csv and get network statistics.





