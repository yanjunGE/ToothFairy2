# **Adaptation de Modèle Med-SAM Adapter aux images CBCT dentaires**

Ce projet permet d'entraîner **Med-SAM Adapter** sur des images CBCT dentaires en utilisant deux types de découpes : **2.5D** et **2D**. 

---

## **1. Deux modes de prétraitement des images**
### **🟢 Option 1 : Utilisation de la découpe 2.5D (par défaut)**
👉 **Processus :** Téléchargez simplement le code depuis GitHub et placez les données d'entraînement dans le dossier `data/`, puis suivez le tutoriel `quick_start.ipynb`.

### **🟢 Option 2 : Utilisation de la découpe 2D**
👉 **Processus :** Remplacez certains fichiers pour permettre un entraînement avec des coupes 2D.

📌 **Fichiers à remplacer dans le répertoire principal :**
- `cfg.py`
- `function.py`
- `start1.ipynb`
- `utils.py`

📌 **Fichiers à placer dans `dataset/` :**
- `samplers.py`
- `toothfairy.py`
- `__init__.py`

📌 **Structure du dataset (ToothFairy) :**
```
data/Dataset/images/...
data/Dataset/labels/...
```

## **2. Installation de l’environnement avec Conda**

### **Étape 1 : Création de l’environnement Conda**
Exécutez les commandes suivantes :
```bash
conda create --name monai-dev python=3.8 -y
conda activate monai-dev
pip install monai timm seaborn opencv-python simpleITK
conda install pytorch torchvision torchaudio -c pytorch
```

### **Étape 2 : Ajouter l’environnement à Jupyter Notebook**
```bash
pip install ipykernel
python -m ipykernel install --user --name=monai-dev --display-name "Python (monai-dev)"
```
Cela permet d’exécuter les notebooks dans l’environnement `monai-dev`.

---

## **3. Lancer l’entraînement**
### **🟢 Pour l’approche 2.5D**
1. Ouvrez `quick_start.ipynb`.
2. Exécutez chaque cellule pour entraîner le modèle.

### **🟢 Pour l’approche 2D**
1. Suivez la section **Option 2 : Utilisation de la découpe 2D** pour remplacer les fichiers nécessaires.
2. Ouvrez `start1.ipynb` et exécutez les cellules.

---

Avec ces configurations, vous pouvez entraîner Med-SAM Adapter efficacement sur des images CBCT dentaires en fonction du type de découpe souhaité. 🚀
