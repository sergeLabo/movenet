

## MoveNet

#### Python
Installe tous les packages nécessaires dans un dossier /mon_env

``` bash
# Mise à jour de pip
python3 -m pip install --upgrade pip
# Installation de venv
sudo apt install python3-venv

# Installation de l'environnement
cd /le/dossier/de/movenet
python3 -m venv mon_env
# Activation
source mon_env/bin/activate
# Installation des packages numpy, opencv, tensorflow
python3 -m pip install -r requirements.txt
```
