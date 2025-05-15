# VERS-Models

[![PyPI - Version](https://img.shields.io/pypi/v/vers-models.svg)](https://pypi.org/project/vers-models)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vers-models.svg)](https://pypi.org/project/vers-models)


## Table des matières
1. [Description](#description)
2. [Installation](#installation)
   1. [PyPi](#pypi)
   2. [Développement](#développement)
3. [Usage](#usage)
   1. [Ligne de commande](#ligne-de-commande)
4. [License](#license)


## Description

## Installation
### PyPi
VERS est disponible sur PyPi, vous pouvez l'installer avec pip à l'aide de la commande suivante:
```bash
pip install vers-models
```
Vous pouvez ensuite vérifier que l'installation s'est bien passée en lançant la commande `vers --version`

### Développement
Pour installer VERS en mode développement, vous pouvez cloner le dépôt git et installer les dépendances avec pip:
```bash
git clone https://github.com/Marceau-H/VERS-Models.git
cd VERS-Models
pip install -e .
```


## Usage
### Ligne de commande
La cli se lance avec `vers` dans un terminal
#### Exemple d'usage:
```bash
vers --train --lang_name metrique_strophe --nb_predictions 100 --model_class multi_head_attn --num_epochs 50 --batch_size 650
```
Cette commande va entraîner (`--train`) un modèle de type `multi_head_attn` (`--model_class`) sur le dataset `metrique_strophe` (`--lang_name`) pendant 50 epochs (`--num_epochs`) avec une taille de batch de 650 (`--batch_size`) et va faire 100 prédictions à la fin de l'entraînement (`--nb_predictions`).

#### Options disponibles:
Cependant, il existe de nombreuses options disponibles pour la commande `vers`, vous pouvez les lister en lançant la commande suivante:
```bash
vers --help
```
Cela vous donnera une liste de toutes les options disponibles et de leur description.

##### Options spécifiques à un modèle:
Les modèles ont également des options qui leur sont propres, en effer, selon l'architecture choisie, différents hyperparamètres peuvent être modifiés. Par exemple, pour le modèle `multi_head_attn`, vous pouvez modifier le nombre de têtes d'attention avec l'option `--num_heads`. Les options disponibles pour chaque modèle sont listées dans le fichier de configuration correspondant dans le dossier `config`, certaines options à spécifier obligatoirement (notifiées par `TO SPECIFY`) et d'autres ont des valeurs par défaut.
Si vous voulez lister les options disponibles pour un modèle, vous pouvez le faire en lançant la commande suivante:
```bash
vers --model_help [model_class]
```
Cela vous donnera une liste de toutes les options disponibles pour le modèle (`model_class` à remplacer par le nom du modèle) et de leur type (défaut, obligatoire, etc.).
###### Exemple d'usage:
```bash
vers --model_help multi_head_attn
```


## License
Ce projet est sous licence AGPL-3.0+ voir le fichier [LICENSE](https://github.com/Marceau-H/VERS/blob/master/LICENSE.txt) pour plus de détails.
