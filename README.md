# LLM Guardrails Bench

Un framework pour tester et évaluer les guardrails anti-toxicité sur le modèle Llama-3.2-1B.

## Description

Ce projet fournit des outils pour tester et évaluer différentes configurations de guardrails anti-toxicité avec le modèle Llama-3.2-1B de Meta. Il permet de mesurer le taux d'activation, la latence, les raisons d'activation et la résistance aux techniques de prompt injection (jailbreak).

Le projet utilise la bibliothèque `guardrails-ai` qui fournit des outils flexibles pour la validation des entrées et sorties des modèles de langage.

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/votre-username/LLM_Guardrails_Bench.git
cd LLM_Guardrails_Bench
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Installer le guardrail anti-toxicité :
```bash
guardrails hub install hub://guardrails/toxic_language
```

## Utilisation

### Tester un guardrail avec des prompts standards

Pour tester le guardrail anti-toxicité avec une configuration standard et un ensemble de prompts prédéfinis :

```bash
python src/test_guardrails.py --threshold 0.5 --method sentence
```

Options disponibles :
- `--prompts` : Chemin vers le fichier contenant les prompts de test (par défaut : `data/test_prompts.txt`)
- `--output` : Chemin pour le fichier de résultats (par défaut : `data/results.csv`)
- `--threshold` : Seuil de confiance pour la toxicité (par défaut : 0.5)
- `--method` : Méthode de validation (`sentence` ou `full`, par défaut : `sentence`)

### Comparer différentes configurations de guardrails

Pour comparer différentes configurations de guardrails (seuils, méthodes de validation) :

```bash
python src/compare_guardrails.py --output-dir data/comparaison
```

Options disponibles :
- `--prompts` : Chemin vers le fichier contenant les prompts de test (par défaut : `data/test_prompts.txt`)
- `--output-dir` : Répertoire de sortie pour les résultats et visualisations (par défaut : `data/comparison`)

### Tester la résistance aux attaques de prompt injection

Pour évaluer la résistance du guardrail aux techniques de prompt injection (jailbreak) :

```bash
python src/test_prompt_injection.py --threshold 0.5 --method sentence
```

Options disponibles :
- `--output` : Chemin pour le fichier de résultats (par défaut : `data/injection_results.csv`)
- `--threshold` : Seuil de confiance pour la toxicité (par défaut : 0.5)
- `--method` : Méthode de validation (`sentence` ou `full`, par défaut : `sentence`)

## Structure du projet

```
.
├── data/                  # Données et résultats
│   ├── test_prompts.txt   # Prompts de test
│   └── ...                # Fichiers de résultats générés
├── src/                   # Code source
│   ├── test_guardrails.py            # Test du guardrail standard
│   ├── compare_guardrails.py         # Comparaison de différentes configurations
│   └── test_prompt_injection.py      # Test de résistance aux prompt injections
├── requirements.txt       # Dépendances Python
└── README.md              # Ce fichier
```

## Fonctionnalités

- **Test de base** : Évaluation simple d'un guardrail avec un ensemble de prompts.
- **Comparaison de configurations** : Comparaison de différents seuils et méthodes de validation.
- **Test de robustesse** : Évaluation de la résistance aux techniques de prompt injection.
- **Métriques détaillées** : Taux d'activation, latence, raisons d'activation.
- **Visualisations** : Graphiques comparatifs des performances des différentes configurations.

## Métriques collectées

- **Taux d'activation** : Pourcentage de réponses où le guardrail a été activé.
- **Temps de génération** : Temps nécessaire pour générer la réponse du modèle.
- **Temps de validation** : Temps nécessaire pour appliquer le guardrail.
- **Raisons d'activation** : Motifs pour lesquels le guardrail a été activé.
- **Résistance aux attaques** : Efficacité contre différentes techniques de prompt injection.

## Dépendances principales

- transformers
- torch
- guardrails-ai
- nltk
- detoxify
- pandas
- matplotlib
- tqdm
- scikit-learn

## Notes importantes

1. Ce projet nécessite un GPU pour fonctionner efficacement avec Llama-3.2-1B.
2. Les guardrails anti-toxicité peuvent parfois produire des faux positifs ou des faux négatifs.
3. Les performances peuvent varier en fonction de la langue et du contexte culturel.

## Limitations

- Le guardrail ne comprend pas toujours le contexte complet ou l'intention derrière certains textes.
- Certaines attaques sophistiquées peuvent potentiellement contourner le guardrail.
- L'évaluation se limite à la toxicité textuelle et ne couvre pas d'autres risques comme les hallucinations ou les biais.

## Licence

Ce projet est disponible sous licence MIT. 