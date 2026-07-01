# SecreatAI

SecreatAI est un prototype de jeu autour du transfert de fichiers par encodage IA. Le but est d'entrainer un autoencodeur, d'utiliser ce modele pour transformer un fichier en `.aiz`, puis de challenger un autre joueur qui tente de reconstruire le fichier avec son propre modele ou avec ses propres methodes d'analyse.

> Important : ce projet est une experimentation ludique et pedagogique sur les autoencodeurs, les representations binaires et l'analyse de patterns. Ce n'est pas un outil de chiffrement fiable pour proteger des donnees sensibles.

## Objectif du jeu

Chaque joueur possede un modele d'autoencodeur entraine separement. Un joueur encode un fichier avec son modele, partage uniquement le fichier `.aiz`, puis l'adversaire tente de le decoder.

Le joueur qui encode marque des points si l'adversaire n'arrive pas a reconstruire le fichier. L'adversaire marque des points s'il parvient a produire une reconstruction correcte.

## Fonctionnalites

- Interface graphique Tkinter pour entrainer, encoder et decoder.
- Entrainement ou reprise d'entrainement d'un modele `.pkl`.
- Encodage de fichiers en `.aiz`.
- Decodage vers le dossier `decoded/`.
- Courbe de loss d'entrainement dans l'interface.
- Outil d'analyse de patterns entre deux fichiers `.aiz`.
- Scripts separes pour l'entrainement, l'encodage et le decodage.

## Prerequis

- Python 3.10, configure par `.python-version`.
- `pip`.
- `tkinter` pour l'interface graphique.

Sur certaines distributions Linux, `tkinter` doit etre installe via le gestionnaire de paquets du systeme, par exemple :

```bash
sudo apt install python3-tk
```

## Installation

Depuis le dossier du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLCONFIGDIR=.matplotlib-cache
```

Les dependances Python actuelles sont volontairement legeres :

- `numpy`
- `matplotlib`

Le projet utilise aussi des modules de la bibliotheque standard Python, notamment `pickle`, `lzma`, `hashlib`, `tarfile`, `threading` et `tkinter`.

## Lancement rapide

L'interface graphique est le moyen le plus simple d'utiliser le projet :

```bash
python SecreatAI_GUI.py
```

Dans l'interface :

1. Choisir un modele existant ou cliquer sur `New Model`.
2. Cliquer sur `Train / Resume` pour creer ou continuer l'entrainement du modele.
3. Choisir un fichier avec `Browse`.
4. Cliquer sur `Encode` pour produire un fichier `.aiz`.
5. Cliquer sur `Decode` pour reconstruire un fichier depuis un `.aiz`.

Les fichiers decodes sont ecrits dans `decoded/`.

## Utilisation en scripts

Les scripts principaux peuvent aussi etre appeles depuis Python.

### Entrainer un modele

```bash
python Training_only.py
```

Par defaut, ce script cree ou reprend `model.pkl`.

Pour utiliser un autre nom de modele :

```bash
python -c "import Training_only; Training_only.model_name='alice.pkl'; Training_only.main()"
```

### Encoder un fichier

```bash
python -c "import Encode_only; Encode_only.main('model.pkl', 'mon_fichier.pdf')"
```

Si la reconstruction interne atteint 100 %, un fichier `mon_fichier.pdf.aiz` est cree dans le meme dossier que le fichier source, sauf si un dossier de sortie est fourni.

Exemple avec dossier de sortie :

```bash
python -c "import Encode_only; Encode_only.main('model.pkl', 'mon_fichier.pdf', output_dir='encoded')"
```

### Decoder un fichier `.aiz`

```bash
python -c "import Decode_only; Decode_only.main('model.pkl', 'mon_fichier.pdf.aiz')"
```

Par defaut, le resultat est ecrit dans `decoded/` avec le suffixe `.aiz` retire.

### Comparer deux fichiers `.aiz`

```bash
python pattern_aiz_compare.py fichier1.aiz fichier2.aiz
```

Ce script affiche des statistiques sur les blocs de 64 bits, les repetitions, l'entropie et la distance de Hamming. Il sert a reperer si des patterns visibles restent presents dans les fichiers encodes.

## Regles proposees

1. Chaque joueur entraine son propre modele `.pkl`.
2. Le joueur actif choisit un fichier source.
3. Il encode le fichier en `.aiz` et verifie que son propre modele peut reconstruire le fichier.
4. Il transmet uniquement le `.aiz` a l'adversaire.
5. L'adversaire tente de decoder le fichier avec son modele ou avec toute autre strategie d'analyse.
6. Le fichier original est revele.
7. Les points sont attribues :
   - adversaire gagnant si le fichier est correctement reconstruit ;
   - encodeur gagnant si le fichier reste indechiffrable.
8. Les roles tournent au round suivant.

## Structure des fichiers

```text
.
|-- SecreatAI_GUI.py                 # Interface graphique
|-- Training_only.py                 # Entrainement/reprise d'un modele
|-- Encode_only.py                   # Encodage d'un fichier vers .aiz
|-- Decode_only.py                   # Decodage d'un .aiz
|-- Autoencoder_Encoder_Decoder.py   # Version combinee historique
|-- Autoencoder_Encoder_Decoder_timed.py
|-- pattern_aiz_compare.py           # Analyse de patterns entre .aiz
|-- Compare_files.py                 # Comparaison simple de fichiers binaires
|-- requirements.txt
|-- model.pkl                        # Modele par defaut, si present
`-- decoded/                         # Sorties decodees
```

## Fichiers generes

- `*.pkl` : modeles entraines.
- `*.aiz` : fichiers encodes.
- `decoded/` : fichiers reconstruits.
- `.matplotlib-cache/` : cache Matplotlib local.
- `__pycache__/` : cache Python.

Ces fichiers sont ignores par `.gitignore` lorsqu'ils sont temporaires ou generes.

## Notes techniques

Le modele travaille sur des blocs de 8 bits en entree et produit des representations encodees de 64 bits. L'encodage est ensuite compresse avec `lzma`, puis masque avec un sel aleatoire avant ecriture du `.aiz`.

L'entrainement utilise une implementation maison de reseau dense et de l'optimiseur Adam avec `numpy`. Les losses d'entrainement et de validation sont stockees dans le fichier `.pkl`, ce qui permet a l'interface d'afficher l'historique.

## Limites connues

- Le format `.aiz` n'est pas un chiffrement cryptographique.
- Un meme modele peut laisser apparaitre des regularites exploitables selon les fichiers et le niveau d'entrainement.
- Les scripts CLI n'ont pas encore d'interface `argparse`; les appels avances se font donc via `python -c` ou via la GUI.
- `pickle` ne doit pas etre utilise avec des modeles non fiables : charger un `.pkl` provenant d'une source inconnue peut executer du code arbitraire.

## Depannage

### Erreur liee a Matplotlib

Definir un dossier de cache local :

```bash
export MPLCONFIGDIR=.matplotlib-cache
```

### Erreur `No module named tkinter`

Installer le paquet systeme Tkinter correspondant a votre Python. Sur Debian/Ubuntu :

```bash
sudo apt install python3-tk
```

### Le decodage produit un fichier incorrect

Verifier que le `.aiz` a ete decode avec le meme modele que celui utilise pour l'encodage. Dans le jeu, utiliser un autre modele fait partie du challenge, mais la reconstruction exacte n'est alors pas garantie.

## Licence

Ce projet est distribue sous licence MIT. Voir [LICENSE](LICENSE).
