# rl-trade-execution
Optimal trade execution framework using reinforcement learning, trained on GARCH-simulated cryptocurrency price data. Minimizes market impact and price risk when executing large orders in volatile markets.

# Optimal Trade Execution with Reinforcement Learning

## Aperçu du projet

Ce projet implémente une approche d'apprentissage par renforcement (RL) pour l'exécution optimale d'ordres de grande taille sur les marchés de cryptomonnaies. L'objectif est de minimiser l'impact de marché et maximiser la valeur obtenue lors de l'exécution d'un ordre important, en adaptant la stratégie aux conditions de marché en temps réel.

## Problématique

Lorsqu'un trader exécute un ordre de grande taille, il fait face à un dilemme:
- Exécuter rapidement risque de créer un fort impact de marché et détériorer le prix
- Exécuter lentement expose à un risque de marché (le prix peut évoluer défavorablement)

Notre projet vise à trouver le compromis optimal entre ces deux risques en utilisant l'apprentissage par renforcement.

## Méthode

Notre approche se distingue par:

1. **Modélisation GARCH du marché**: Nous utilisons un modèle GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) calibré sur des données réelles pour simuler des trajectoires de prix réalistes qui capturent le phénomène crucial de "clustering de volatilité".

2. **Entraînement sur données simulées**: Pour éviter le problème de rareté des données et permettre une exploration sûre, notre agent RL est entraîné sur des trajectoires de prix simulées.

3. **Validation sur données réelles**: Les performances sont ensuite évaluées sur des données de marché historiques réelles.

## Composants implémentés

### 1. Collecte et préparation des données
- Collecteur de données Binance pour télécharger l'historique BTCUSDT
- Prétraitement et ingénierie de caractéristiques

### 2. Modèle GARCH et simulation
- Calibration du modèle GARCH(1,1) sur les données historiques
- Générateur de trajectoires de prix qui reproduit les propriétés statistiques essentielles du marché
- Calcul de la volatilité réalisée comme mesure cohérente entre données simulées et réelles

## Fondements théoriques et choix de conception

### Modèle GARCH

#### Formulation mathématique
Le modèle GARCH (Generalized AutoRegressive Conditional Heteroskedasticity) est une extension des modèles ARCH développée par Bollerslev (1986). Dans sa forme GARCH(1,1), le modèle est défini par:

Pour les rendements:
$$r_t = \mu + \sigma_t \cdot Z_t$$

Pour la variance conditionnelle:
$$\sigma_t^2 = \omega + \alpha \cdot r_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$$

Où:
- $r_t$ représente le rendement à l'instant $t$
- $\mu$ est la moyenne des rendements (généralement proche de zéro pour des données à haute fréquence)
- $\sigma_t$ est la volatilité conditionnelle à l'instant $t$
- $Z_t$ est un bruit blanc standardisé, généralement distribué selon une loi normale $N(0,1)$
- $\omega, \alpha, \beta$ sont les paramètres à estimer, avec $\omega > 0$, $\alpha, \beta \geq 0$ et $\alpha + \beta < 1$ pour garantir la stationnarité

#### Calibration
L'estimation des paramètres est réalisée par maximum de vraisemblance:
1. Les rendements sont centrés ($r_t - \mu$)
2. La log-vraisemblance est définie comme:
   $$L(\omega, \alpha, \beta) = -\frac{1}{2}\sum_{t=1}^T \left(\log(\sigma_t^2) + \frac{(r_t-\mu)^2}{\sigma_t^2}\right)$$
3. Les paramètres optimaux sont ceux qui maximisent cette log-vraisemblance

Nous utilisons 5000 points de données pour la calibration, ce qui offre un compromis entre:
- Précision statistique (suffisamment de données)
- Pertinence locale (capture le régime de marché récent)
- Efficacité computationnelle (temps de calcul raisonnable)

#### Simulation de trajectoires
Pour générer des trajectoires de prix à partir du modèle calibré:

1. Initialisation:
   - Variance initiale: $\sigma_0^2 = \omega/(1-\alpha-\beta)$ (variance inconditionnelle)
   - Prix initial: dernière valeur observée $P_0$

2. Pour chaque pas de temps $t = 1, 2, \ldots$:
   - Générer $Z_t \sim N(0,1)$
   - Calculer $\sigma_t^2 = \omega + \alpha \cdot r_{t-1}^2 + \beta \cdot \sigma_{t-1}^2$
   - Calculer $r_t = \mu + \sigma_t \cdot Z_t$
   - Calculer le nouveau prix: $P_t = P_{t-1} \cdot \exp(r_t)$

#### Volatilité réalisée
Pour assurer la cohérence entre environnement simulé et réel, nous utilisons une mesure de volatilité réalisée calculée identiquement dans les deux contextes:

$$\text{volatilité réalisée}_t = \sqrt{\frac{1}{n-1}\sum_{i=t-n+1}^{t} (r_i - \bar{r})^2}$$

Où:
- $n$ est la taille de la fenêtre (typiquement 20 observations)
- $r_i$ sont les rendements logarithmiques
- $\bar{r}$ est la moyenne des rendements dans la fenêtre

### Absence de momentum directionnel
Dans notre implémentation GARCH, il n'y a pas de momentum directionnel dans les prix simulés pour plusieurs raisons:

1. **Cohérence avec la théorie financière**: À l'échelle intra-journalière, les marchés liquides comme BTC/USDT montrent peu d'autocorrélation dans les rendements (efficience de forme faible)

2. **Éviter le sur-apprentissage**: Un agent RL entraîné sur des données avec momentum artificiel pourrait apprendre des stratégies qui ne fonctionnent pas sur le marché réel

3. **Robustesse du modèle**: GARCH(1,1) capture efficacement le clustering de volatilité sans introduire de dépendance sérielle dans les rendements

Cette approche a été validée empiriquement en comparant les distributions des rendements simulés et réels, confirmant que notre modèle reproduit fidèlement les propriétés statistiques essentielles.

### Structure du state pour le RL
Après analyse, nous avons identifié les variables d'état les plus informatives pour l'agent RL:

1. **Inventaire normalisé** (quantité restant à vendre / quantité totale)
2. **Temps restant normalisé** (temps restant / horizon total)
3. **Volatilité réalisée** (écart-type des rendements sur une fenêtre glissante)
4. **Prix relatif** (prix actuel / prix initial)

Cette représentation minimaliste mais complète évite d'inclure des features artificielles qui ne seraient pas informatives dans un modèle GARCH pur.

## Prochaines étapes

1. **Développement de l'environnement RL**:
   - Implémentation d'un environnement compatible Gymnasium
   - Modélisation de l'impact de marché proportionnel à la volatilité
   - Conception de la fonction de récompense

2. **Implémentation de l'agent**:
   - Architecture DQN ou PPO pour l'apprentissage
   - Paramétrisation des actions (pourcentage de l'inventaire à exécuter)
   - Stratégies d'exploration adaptées

3. **Évaluation et optimisation**:
   - Comparaison avec des stratégies de référence (TWAP, VWAP)
   - Analyse de sensibilité aux paramètres du modèle
   - Tests sur données de marché out-of-sample

## Structure du projet

```
rl-trade-execution/
├── data/                 # Données brutes et traitées
├── src/
│   ├── data/             # Modules de collecte et traitement
│   ├── environment/      # Simulateur GARCH et environnement RL  
│   └── models/           # Implémentations des agents RL
├── scripts/              # Scripts d'analyse et de test
├── sample/               # Exemples de données simulées
└── notebooks/            # Analyses exploratoires
```