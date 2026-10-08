

---

## 1. Download data
- **PH $^2$** dataset: https://www.fc.up.pt/addi/ph2%20database.html
- **Derm7pt** dataset: https://derm.cs.sfu.ca/Welcome.html
- **HAM10000** dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

1.1 After downloading data, the directory structure of `data` folder should look like this:
```bash
.
└── data/
    ├── PH2/
    │   ├── images/
    │   │   ├── IMD002.jpg
    │   │   └── ...
    │   └── splits/
    │       ├── PH2_train_split_0.csv
    │       └── ...
    ├── Derm7pt/
    │   ├── images/
    │   │   └── Aal002.jpg
    │   └── splits/
    │       ├── derm7pt_train.csv
    │       └── ...
    ├── HAM10000/
    │   ├── images/
    │   │   └── ISIC_0024306.jpg
    │   └── splits/
    │       ├── HAM10000_train.csv
    │       └── ...
    └── ...
``` 

You should copy the downloaded images to the `images/` folder in each dataset's folder. If you have any trouble with downloading datasets/images, please e-mail me at *cristiano [dot] patricio [at] ubi [dot] pt*.

## 2. Installation


2.1 Clone this repository and navigate to the folder:
```bash
git clone https://github.com/CristianoPatricio/2-step-concept-based-skin-diagnosis.git
cd 2-step-concept-based-skin-diagnosis/
```

2.2 Create a new conda environment and install required libraries contained in `requirements.txt` file:

```bash
conda create --name 2-step-skin python=3.10
conda activate 2-step-skin
pip install -r requirements.txt
```



**Note:** We recommend using a GPU with a minimum of 64 GB of memory.

## 3. Reproduce experiments

**Note:** For experiments with ExpLICD, you need to clone the `Explicd` repository into `2-step-concept-based-skin-diagnosis/src/` folder.

```bash
cd src/
git clone https://github.com/yhygao/Explicd.git
```

Then, download the .pth checkpoint of the pretrained ExpLICD on ISIC 2018 from [Google Drive](https://drive.google.com/file/d/1jl33-St8ksbivpE5t5PSsrVBL49p4pwU/view?usp=share_link) and move it to the `checkpoints` folder.

For the experiments with CBI-VLM, refer to the [this repository](https://github.com/CristianoPatricio/concept-based-interpretability-VLM).

3.1 [Table 4] Predict class label from image features ($x \rightarrow y$)
```bash
./scripts/run_x_to_y.sh 0    # GPU ID: 0
```

3.2 Few-shot disease classification

**Note:** You'll need to download the pre-computed visual features from [Google Drive](https://drive.google.com/file/d/1uZgiHltaCqA2ldMbk7MOZj5jJPt9-uQf/view?usp=sharing) and move it to the `data/visual_features` folder. You could also download the generated concepts ($x \rightarrow c$) for each model from [Google Drive](https://drive.google.com/file/d/1wHdHxMVI8eis_V49PnHvIOk9acjrP2dk/view?usp=sharing) and move it to the `data/concept_prediction` folder.

```bash
./scripts/run_x_to_c_to_y.sh 0  # GPU ID: 0
```

3.3 VLM + Linear Classifier

```bash
./scripts/run_vlm_linear_classifier.sh 0  # GPU ID: 0
```


"# MedicalConcept" 
