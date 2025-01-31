# **README - Utilisation de 2D Slices pour l'entraînement de MedSAM Adapter**

## **1. Remplacement des fichiers dans le projet GitHub**
Pour utiliser les coupes 2D avec MedSAM Adapter, vous devez remplacer certains fichiers du projet GitHub principal par ceux fournis dans ce dossier :

### **Fichiers à remplacer dans le répertoire principal :**
- `cfg.py`
- `function.py`
- `start1.ipynb`
- `utils.py`

### **Fichiers à placer dans `dataset/` :**
- `samplers.py`
- `toothfairy.py`
- `__init__.py`

## **2. Structure des données (ToothFairy Dataset)**
Assurez-vous que votre dataset ToothFairy est organisé comme suit :
```
data/Dataset/images/...
data/Dataset/labels/...
```

## **3. Initialisation de l’environnement sur Saturn Cloud**
### **Étape 1 : Activer l’environnement `monai-dev`**
Exécutez les commandes suivantes dans le terminal :
```bash
conda env create -f monai-dev.yml
conda activate monai-dev
pip install monai timm seaborn opencv-python simpleITK
sudo apt-get install libglib2.0-0
```

### **Étape 2 : Ajouter l’environnement sur Jupyter Kernel**
```bash
python -m ipykernel install --user --name=monai-dev --display-name "Python (monai-dev)"
```
Cela vous permet d’utiliser cet environnement directement dans Jupyter Notebook sur Saturn Cloud.

---

## **4. Exécution du projet sur Saturn Cloud**
1. **Démarrez un nouvel environnement Saturn Cloud.**
2. **Ouvrez un terminal et exécutez les commandes ci-dessus** pour configurer l’environnement.
3. **Ouvrez `start1.ipynb` et lancez les cellules** pour commencer l’entraînement avec des coupes 2D.

---

Avec cette configuration, vous pourrez entraîner MedSAM Adapter avec des coupes 2D en utilisant Saturn Cloud. 🚀
