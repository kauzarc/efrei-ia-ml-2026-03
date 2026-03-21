# QCM — IA, Deep Learning et Machine Learning

Questions orientées compréhension conceptuelle. Format : 3 options (A/B/C), une seule bonne réponse.

---

## J1AM — Fondations mathématiques & outils Python

**Q1.** Pourquoi préfère-t-on utiliser NumPy plutôt que des boucles Python pour les opérations sur les tableaux ?

A. NumPy produit un code plus lisible et plus court que les boucles Python équivalentes
B. Python ne supporte pas nativement les boucles sur des tableaux numériques multi-dimensionnels
C. Les opérations NumPy sont vectorisées et exécutées en C compilé ✓

> **Réponse : C** — NumPy délègue les calculs à des routines C/Fortran optimisées ; les boucles Python sont interprétées et ont un overhead important par itération.

---

**Q2.** Qu'est-ce que le broadcasting en NumPy ?

A. La distribution automatique des calculs sur plusieurs cœurs CPU en parallèle
B. Un mécanisme pour opérer sur des tableaux de formes différentes en étirant les dimensions ✓
C. La copie implicite d'un tableau en mémoire pour éviter les effets de bord lors des opérations

> **Réponse : B** — Le broadcasting évite de créer des copies inutiles : NumPy traite un vecteur `(n,)` comme s'il était répliqué pour s'aligner sur une matrice `(m, n)`.

---

**Q3.** Quelle est la différence entre `A @ B` et `A * B` pour deux matrices NumPy ?

A. `@` effectue le produit élément par élément, `*` effectue le produit matriciel
B. `@` effectue le produit matriciel, `*` effectue le produit élément par élément ✓
C. Les deux sont équivalents si les matrices ont la même forme carrée

> **Réponse : B** — `A @ B` est le produit matriciel (équivalent à `np.matmul`) : le résultat `(i,j)` est le produit scalaire de la ligne `i` de A avec la colonne `j` de B. `A * B` multiplie chaque élément avec son homologue — les formes doivent être compatibles au sens du broadcasting.

---

**Q4.** Quelle est la différence entre une matrice et un vecteur du point de vue d'une transformation linéaire ?

A. Un vecteur est une liste de scalaires, une matrice est plusieurs vecteurs empilés — sans différence conceptuelle
B. Une matrice encode une transformation linéaire ; un vecteur représente un point ou une direction dans l'espace ✓
C. Les matrices ne peuvent transformer que des vecteurs colonnes, jamais des vecteurs lignes

> **Réponse : B** — Multiplier `Ax` applique la transformation encodée par `A` au vecteur `x`, changeant potentiellement sa direction et sa norme.

---

**Q5.** Qu'est-ce qu'un gradient ?

A. La valeur de la fonction de perte en un point donné de l'espace des paramètres
B. La dérivée seconde d'une fonction scalaire, indiquant sa courbure locale
C. Le vecteur des dérivées partielles d'une fonction scalaire ✓

> **Réponse : C** — Pour descendre une fonction de perte, on se déplace dans la direction opposée au gradient (descente de gradient).

---

**Q6.** Pourquoi la règle de la chaîne (chain rule) est-elle centrale en deep learning ?

A. Elle permet de calculer la dérivée d'une fonction composée couche par couche ✓
B. Elle simplifie les calculs matriciels en décomposant les produits de matrices en facteurs élémentaires
C. Elle garantit la convergence de la descente de gradient vers un minimum global ou local optimal

> **Réponse : A** — Un réseau de neurones est une composition de fonctions (couches) ; la rétropropagation applique la chain rule pour propager les gradients couche par couche.

---

**Q7.** Quelle est la différence entre un DataFrame Pandas et un tableau NumPy ?

A. Un DataFrame a des colonnes nommées et peut contenir des types hétérogènes ; NumPy est homogène ✓
B. Pandas est plus lent que NumPy car il repose sur un interpréteur Python pur sans optimisation native
C. NumPy supporte nativement les données textuelles et catégorielles, contrairement à Pandas

> **Réponse : A** — Pandas est conçu pour les données tabulaires avec métadonnées (noms de colonnes, index, types mixtes) ; NumPy est optimisé pour le calcul numérique homogène.

---

**Q8.** Pourquoi normaliser les features avant d'appliquer la descente de gradient ?

A. Sans normalisation, NumPy ne peut pas effectuer les calculs matriciels en haute dimension correctement
B. Sans normalisation, les features à grande échelle dominent le gradient ✓
C. La normalisation améliore la lisibilité et la reproductibilité du code mais n'a pas d'impact sur l'optimisation

> **Réponse : B** — Si une feature varie de 0 à 1000 et une autre de 0 à 1, le gradient sera très sensible à la première et quasi-insensible à la seconde. Normaliser rend les pas équilibrés dans toutes les directions et accélère la convergence.

---

## J1PM — Principes du ML & Régression linéaire

**Q1.** Quelle est la différence entre apprentissage supervisé, non supervisé et par renforcement ?

A. Supervisé = étiquettes fournies ; non supervisé = pas d'étiquettes ; renforcement = récompenses par interaction ✓
B. Supervisé = données structurées ; non supervisé = données brutes ; renforcement = apprentissage en ligne
C. Ce sont trois paradigmes distincts selon le type de données, mais les algorithmes sous-jacents restent les mêmes

> **Réponse : A** — La distinction clé est la nature du signal d'apprentissage : paires (x, y) étiquetées, structure des données seules, ou signal de récompense retardé.

---

**Q2.** Pourquoi ne peut-on pas évaluer un modèle uniquement sur ses données d'entraînement ?

A. Les données d'entraînement sont généralement trop volumineuses pour calculer des métriques fiables
B. Les métriques sur les données d'entraînement sont biaisées car l'échantillon n'est pas représentatif
C. Un modèle peut mémoriser les exemples d'entraînement sans généraliser (overfitting) ✓

> **Réponse : C** — L'objectif est la généralisation : un modèle qui mémorise son jeu d'entraînement obtient 100% mais échoue sur de nouvelles données.

---

**Q3.** Que représente la fonction de perte (loss function) dans un problème d'apprentissage ?

A. Le temps de calcul total nécessaire pour entraîner le modèle sur un epoch entier
B. Le nombre total de paramètres et de couches du modèle à optimiser pendant l'entraînement
C. Une mesure de l'écart entre les prédictions et les valeurs réelles, à minimiser ✓

> **Réponse : C** — La loss quantifie l'erreur ; l'entraînement consiste à ajuster les paramètres pour réduire cette erreur sur les données d'entraînement.

---

**Q4.** Quel est l'effet d'un taux d'apprentissage (learning rate) trop élevé lors de l'optimisation ?

A. Les mises à jour dépassent le minimum et la loss diverge ou oscille ✓
B. L'entraînement est plus lent car chaque mise à jour nécessite davantage de calculs et d'itérations
C. Le modèle converge plus rapidement et de façon plus stable vers un minimum global plus précis

> **Réponse : A** — Un learning rate trop grand fait « sauter » par-dessus le minimum ; trop petit, la convergence est lente — il faut trouver le bon équilibre.

---

**Q5.** Quelle est la différence entre le gradient descent en batch, stochastique et mini-batch ?

A. Batch = 1 exemple ; stochastique = tout le dataset ; mini-batch = un sous-ensemble
B. Ce sont des variantes d'un même algorithme dont le nom change selon la taille du modèle utilisé
C. Batch = tout le dataset ; stochastique = 1 exemple ; mini-batch = un sous-ensemble ✓

> **Réponse : C** — La taille du lot utilisé pour calculer le gradient change le bruit des mises à jour et l'utilisation mémoire/GPU.

---

**Q6.** Pourquoi utilise-t-on le mini-batch en pratique plutôt que le batch ou le SGD pur ?

A. Le mini-batch est mathématiquement plus précis car il réduit la variance du gradient estimé à chaque itération
B. Le mini-batch est la seule méthode compatible avec la rétropropagation sur des données hétérogènes
C. Il équilibre stabilité du gradient et efficacité GPU sans charger tout le dataset en RAM ✓

> **Réponse : C** — Le mini-batch exploite le parallélisme GPU (tenseurs en batch), introduit du bruit régularisant utile, et reste plus stable que le SGD pur.

---

**Q7.** Pourquoi ne peut-on pas toujours utiliser la solution analytique pour la régression linéaire ?

A. La solution analytique n'existe que pour les problèmes avec une seule variable explicative
B. La solution analytique produit des poids qui ne peuvent pas être interprétés ni comparés entre features
C. Inverser `(XᵀX)` est coûteux en O(p³) et instable si les features sont colinéaires ✓

> **Réponse : C** — L'équation normale `w = (XᵀX)⁻¹Xᵀy` est exacte mais coûteuse (p features) et numériquement instable ; le GD s'adapte mieux aux grandes dimensions.

---

**Q8.** Qu'est-ce que la convexité d'une fonction de perte nous garantit-elle pour l'optimisation ?

A. Que l'algorithme convergera en un nombre fixe et prévisible d'itérations vers la solution exacte
B. Qu'il n'existe pas de minimum local : tout minimum est le minimum global ✓
C. Que le gradient est toujours positif ou nul, ce qui facilite la descente vers la solution optimale

> **Réponse : B** — Pour une fonction convexe (comme la MSE de la régression linéaire), tout minimum local est global — la descente de gradient trouve la solution optimale.

---

## J2AM — Classification & Perceptron

**Q1.** Quelle est la différence fondamentale entre un problème de régression et un problème de classification ?

A. La régression utilise des réseaux de neurones, la classification utilise des arbres de décision
B. La régression prédit une valeur continue ; la classification prédit une catégorie discrète ✓
C. La régression fonctionne uniquement sur des données numériques, la classification sur des données textuelles

> **Réponse : B** — Ex : prédire un prix (régression) vs prédire si un email est spam ou non (classification).

---

**Q2.** Pourquoi la fonction `sign` ne peut-elle pas être utilisée directement dans une descente de gradient ?

A. La fonction sign est discontinue en 0 et trop coûteuse à calculer pour les très grands réseaux
B. Elle n'est pas différentiable : gradient nul partout et indéfini en 0 ✓
C. La fonction sign sature les activations et produit des valeurs trop grandes qui déstabilisent l'entraînement

> **Réponse : B** — La descente de gradient requiert des gradients utilisables ; `sign` a un gradient nul, ce qui bloque toute mise à jour des poids.

---

**Q3.** Quel est le rôle d'une fonction d'activation non-linéaire dans un réseau de neurones ?

A. Elle normalise les activations entre -1 et 1 pour éviter l'explosion des gradients en profondeur
B. Elle permet au réseau d'apprendre des frontières non linéaires ✓
C. Elle convertit les sorties du neurone en probabilités normalisées entre 0 et 1 comme le ferait un softmax

> **Réponse : B** — Sans non-linéarité, `W₂(W₁x) = (W₂W₁)x` : le réseau entier se réduit à une seule transformation linéaire.

---

**Q4.** Pourquoi ReLU est-elle préférée à d'autres fonctions non-linéaires en pratique ?

A. Son gradient est constant pour x>0, ce qui évite le vanishing gradient ✓
B. ReLU est la seule activation différentiable qui garantit la convergence du réseau en classification multi-classe
C. ReLU borne ses sorties entre 0 et 1 à chaque couche, normalisant les activations et accélérant l'entraînement

> **Réponse : A** — Sigmoid et tanh saturent (gradient → 0), bloquant la rétropropagation dans les couches profondes. ReLU n'a pas ce problème pour x>0.

---

**Q5.** Quelle est la limitation fondamentale du perceptron de Rosenblatt à une seule couche ?

A. Il ne peut apprendre que des frontières linéaires, incapable de traiter des données non linéairement séparables ✓
B. Il est trop lent à entraîner pour des grands datasets car chaque mise à jour requiert un passage complet sur les données
C. Il ne fonctionne que pour la classification binaire et ne supporte pas les problèmes multi-classe ou multi-label

> **Réponse : A** — Le théorème de convergence du perceptron garantit qu'il trouve une solution si et seulement si les données sont linéairement séparables.

---

**Q6.** Pourquoi le problème XOR est-il un cas emblématique en deep learning ?

A. XOR est le problème de référence le plus difficile en optimisation combinatoire discrète
B. Il démontre l'échec du perceptron simple sur XOR et la capacité du MLP à le résoudre ✓
C. XOR est utilisé comme fonction d'initialisation des poids dans les réseaux profonds modernes

> **Réponse : B** — Minsky & Papert (1969) ont utilisé XOR pour montrer les limites du perceptron ; l'ajout d'une couche cachée suffit à le résoudre.

---

**Q7.** À quoi sert la fonction softmax dans une classification multi-classe ?

A. Elle convertit un vecteur de logits en distribution de probabilités qui somme à 1 ✓
B. Elle sélectionne et amplifie la classe avec le score le plus élevé, supprimant les autres
C. Elle normalise les activations de la dernière couche pour stabiliser et accélérer l'entraînement

> **Réponse : A** — Softmax(`zᵢ`) = `exp(zᵢ) / Σ exp(zⱼ)` ; chaque sortie est entre 0 et 1 et la somme vaut 1 — interprétable comme une probabilité.

---

**Q8.** Que signifie que la sortie du softmax est une distribution de probabilité ?

A. Les sorties sont triées par ordre décroissant et la plus grande indique la classe prédite
B. Le réseau garantit que sa prédiction est correcte avec la probabilité indiquée par la sortie
C. Chaque valeur est entre 0 et 1 et leur somme vaut 1 — interprétable comme probabilité ✓

> **Réponse : C** — C'est une propriété mathématique du softmax, pas une garantie de calibration ; un modèle peut être confiant et faux.

---

**Q9.** Pourquoi ajouter des couches cachées avec des non-linéarités permet-il de résoudre XOR ?

A. La couche cachée transforme l'espace des features en une représentation linéairement séparable ✓
B. Les couches supplémentaires augmentent la capacité du modèle, lui permettant de mémoriser les patterns complexes
C. Les non-linéarités introduisent une régularisation implicite qui aide à éviter l'overfitting sur XOR

> **Réponse : A** — La couche cachée réorganise l'espace d'entrée ; dans la nouvelle représentation, XOR devient linéairement séparable.

---

## J2PM — Validation & Généralisation

**Q1.** Comment distingue-t-on l'overfitting de l'underfitting en pratique ?

A. Overfitting = loss d'entraînement élevée et instable ; underfitting = loss d'entraînement faible et stable
B. Overfitting = modèle trop petit avec capacité insuffisante ; underfitting = modèle trop grand avec trop de paramètres
C. Overfitting = faible loss train mais forte loss validation ; underfitting = forte loss sur les deux ✓

> **Réponse : C** — L'overfitting se détecte par l'écart entre les courbes train et validation ; l'underfitting se voit par de mauvaises performances partout.

---

**Q2.** Pourquoi un modèle avec 100% de précision en entraînement n'est-il pas forcément bon ?

A. 100% de précision indique un bug dans le code d'entraînement ou un label leakage
B. La précision n'est pas une métrique fiable ; il faut toujours privilégier la F1-score sur les données d'entraînement
C. Le modèle a peut-être mémorisé les données sans apprendre les patterns généralisables (overfitting) ✓

> **Réponse : C** — Un modèle qui mémorise les exemples d'entraînement atteint 100%, mais ses performances sur de nouvelles données peuvent être médiocres.

---

**Q3.** À quoi sert le jeu de validation, distinct du jeu de test ?

A. Le jeu de validation sert aux itérations intermédiaires ; le jeu de test à l'entraînement final du modèle retenu
B. Le jeu de validation est plus petit que le jeu de test pour maximiser les données d'entraînement disponibles
C. Le jeu de validation sert à sélectionner les hyperparamètres sans contaminer le jeu de test ✓

> **Réponse : C** — Utiliser le test set pour choisir un modèle revient à optimiser dessus ; le test set doit rester « invisible » jusqu'à l'évaluation finale.

---

**Q4.** Qu'est-ce que le compromis biais-variance ?

A. Le choix entre un modèle rapide à l'inférence (biais élevé) et un modèle plus précis mais lent (variance élevée)
B. Le biais mesure l'erreur systématique sur l'entraînement, la variance mesure l'erreur aléatoire sur le test
C. Un modèle complexe a peu de biais mais forte variance ; un modèle simple a l'inverse ✓

> **Réponse : C** — Erreur = Biais² + Variance + Bruit irréductible ; trouver la bonne complexité du modèle minimise cette somme.

---

**Q5.** Un modèle trop simple souffre-t-il plutôt de biais ou de variance ?

A. De variance élevée, car il ne peut pas capturer la structure des données et généralise mal
B. Ni l'un ni l'autre — un modèle simple est toujours préférable selon le principe du rasoir d'Occam
C. De biais élevé : il fait des hypothèses trop restrictives sur la relation features-cible ✓

> **Réponse : C** — Un modèle trop simple (ex : régression linéaire sur des données non linéaires) sous-ajuste systématiquement : c'est du biais (erreur structurelle).

---

**Q6.** Pourquoi utilise-t-on la validation croisée (k-fold) plutôt qu'un simple split train/test ?

A. La k-fold est plus rapide car elle évite de réentraîner le modèle sur chaque fold séparément
B. Elle donne une estimation plus robuste en faisant tourner la validation sur chaque fold ✓
C. La k-fold garantit que le modèle ne fait jamais d'overfitting sur les données d'entraînement

> **Réponse : B** — Avec un seul split, l'estimation dépend du hasard du split ; la k-fold moyenne sur k splits, donnant une estimation plus fiable et un intervalle de confiance.

---

**Q7.** Quelle est la différence entre un hyperparamètre et un paramètre appris ?

A. Les hyperparamètres (learning rate, nb de couches) sont des paramètres internes mis à jour par gradient ; les paramètres appris sont fournis avant l'entraînement
B. Il n'y a pas de différence fonctionnelle : les deux types sont optimisés par descente de gradient et mis à jour à chaque itération
C. Les paramètres appris sont optimisés par descente de gradient ; les hyperparamètres sont fixés avant l'entraînement ✓

> **Réponse : C** — Les hyperparamètres contrôlent le processus d'apprentissage et ne sont pas mis à jour par rétropropagation ; on les choisit par validation croisée ou grid search.

---

**Q8.** Pourquoi ne doit-on jamais utiliser le jeu de test pour sélectionner un modèle ?

A. Le jeu de test est trop petit pour des métriques fiables (biais d'échantillonnage élevé sur des données peu représentatives)
B. Cela revient à s'adapter au test set, rendant l'évaluation finale optimiste et non représentative ✓
C. Les métriques sur le test set sont moins stables que sur le validation set car il est plus petit et moins diversifié

> **Réponse : B** — Si on choisit le meilleur modèle selon le test set, on optimise indirectement dessus ; l'estimation des performances en production sera biaisée.

---

## J3AM — Autograd & Rétropropagation

**Q1.** Pourquoi a-t-on besoin de la rétropropagation pour entraîner un réseau profond ?

A. Elle calcule le gradient de la loss par rapport à chaque paramètre en appliquant la chain rule ✓
B. Elle accélère le forward pass en mémorisant les activations intermédiaires pour les réutiliser à l'itération suivante
C. Elle permet de paralléliser l'entraînement sur plusieurs GPUs en synchronisant les gradients entre processus

> **Réponse : A** — Sans rétropropagation, calculer ∂L/∂wᵢ pour chaque poids serait prohibitivement coûteux ; backprop le fait en un seul passage arrière.

---

**Q2.** Qu'est-ce que le "problème de l'attribution du crédit" (credit assignment problem) ?

A. La difficulté à identifier quels neurones sont responsables d'une erreur dans les couches profondes ✓
B. La difficulté à répartir équitablement les données d'entraînement entre toutes les couches du réseau
C. Les conflits entre gradients contradictoires qui s'appliquent simultanément au même poids lors du backward

> **Réponse : A** — Chaque couche contribue à l'erreur finale ; backprop résout ce problème en propageant les gradients couche par couche depuis la sortie.

---

**Q3.** Qu'est-ce qu'un graphe de calcul (computation graph) ?

A. Un graphique montrant la progression de la loss et des métriques au fil des epochs d'entraînement
B. Un graphe acyclique dirigé (DAG) des opérations mathématiques d'un modèle ✓
C. Un diagramme de l'architecture du réseau montrant les couches, les connexions et les dimensions

> **Réponse : B** — Le graphe de calcul trace toutes les opérations effectuées lors du forward pass, ce qui permet de calculer automatiquement les gradients en le parcourant à l'envers.

---

**Q4.** Quelle est la différence entre la passe avant (forward pass) et la passe arrière (backward pass) ?

A. Forward = entraînement sur les données ; backward = inférence sur de nouvelles entrées
B. Forward = calcul de la prédiction et de la loss ; backward = calcul des gradients de la sortie vers l'entrée ✓
C. Forward = calcul des gradients couche par couche ; backward = mise à jour des poids par descente de gradient

> **Réponse : B** — Ces deux passes constituent une itération d'entraînement : forward pour calculer la loss, backward pour calculer les gradients, puis mise à jour des paramètres.

---

**Q5.** Pourquoi empiler des couches linéaires sans non-linéarités entre elles n'a aucun intérêt ?

A. Composer des couches linéaires reste linéaire : tout le réseau équivaut à une seule transformation ✓
B. Les couches linéaires en séquence génèrent une explosion de la mémoire GPU lors du calcul matriciel
C. Les gradients s'annulent lors de la rétropropagation à travers plusieurs couches linéaires successives

> **Réponse : A** — Peu importe le nombre de couches linéaires empilées, le modèle reste une régression linéaire ; les non-linéarités sont indispensables pour l'expressivité.

---

**Q6.** Comment les gradients se propagent-ils lorsqu'un nœud a plusieurs chemins entrants ?

A. Seul le chemin avec le gradient le plus élevé est conservé pour éviter les conflits entre directions
B. Les gradients des différents chemins s'additionnent au nœud ✓
C. Les gradients des différents chemins sont multipliés entre eux pour amplifier le signal

> **Réponse : B** — Dans backprop, quand un nœud a plusieurs sorties (ou reçoit des gradients depuis plusieurs chemins), on additionne les contributions — c'est la règle des dérivées partielles.

---

**Q7.** Pourquoi faut-il remettre les gradients à zéro (`zero_grad`) avant chaque itération d'entraînement ?

A. Pour libérer la mémoire GPU allouée aux activations intermédiaires stockées pendant le forward pass
B. Sans remise à zéro, les gradients s'accumulent entre batches et faussent les mises à jour ✓
C. Pour empêcher les gradients de dépasser 1 et éviter une explosion numérique lors du backward

> **Réponse : B** — Ce comportement d'accumulation est intentionnel (utile pour les gradients accumulés sur plusieurs batches), mais doit être géré explicitement à chaque itération normale.

---

## J3PM — PyTorch & CNN

**Q1.** Quelle est la différence entre un tenseur PyTorch et un tableau NumPy ?

A. Les tenseurs PyTorch supportent GPU et autograd ; NumPy est CPU uniquement et sans différentiation ✓
B. Les tenseurs PyTorch ne supportent que les entiers ; NumPy supporte les types flottants et complexes
C. NumPy est plus rapide que PyTorch pour les opérations numériques car il évite l'overhead du graphe de calcul

> **Réponse : A** — PyTorch étend NumPy avec le support GPU et l'autograd, essentiels pour entraîner des réseaux de neurones efficacement.

---

**Q2.** À quoi sert le flag `requires_grad` dans PyTorch ?

A. Il indique que le tenseur doit être copié avant toute opération pour éviter les modifications en place
B. Il force le tenseur à rester sur CPU même si un GPU est disponible, pour la reproductibilité
C. Il signale à PyTorch de suivre ce tenseur dans le graphe de calcul pour calculer son gradient ✓

> **Réponse : C** — Seuls les tenseurs avec `requires_grad=True` (les poids du modèle) accumulent les gradients ; les données d'entrée n'en ont pas besoin.

---

**Q3.** Pourquoi un MLP (réseau dense) est-il peu adapté au traitement d'images ?

A. Les MLP ne supportent pas nativement les entrées 2D et requièrent une conversion préalable en vecteur 1D
B. Les MLP génèrent un nombre de paramètres explosif et n'ont pas d'invariance aux translations ✓
C. Les MLP produisent des sorties continues non bornées, inadaptées à la classification d'images en catégories

> **Réponse : B** — Une image 224×224×3 donne ~150k entrées ; avec un MLP, la première couche seule aurait des millions de paramètres, sans exploiter la structure spatiale.

---

**Q4.** Qu'est-ce que le partage de poids dans une convolution ?

A. Le même filtre est appliqué à toutes les positions de l'image, réduisant le nombre de paramètres ✓
B. Les poids sont partagés entre plusieurs réseaux entraînés en parallèle pour accélérer l'entraînement distribué
C. Les couches adjacentes d'un CNN partagent leurs poids pour réduire la mémoire GPU nécessaire à l'entraînement

> **Réponse : A** — Un filtre de taille 3×3×3 a seulement 27 paramètres quelle que soit la taille de l'image ; il détecte le même pattern où qu'il apparaisse.

---

**Q5.** Quelle est la différence entre stride et padding dans une couche de convolution ?

A. Le stride contrôle la taille du filtre de convolution ; le padding contrôle le nombre de filtres appliqués
B. Stride et padding sont deux noms pour le même hyperparamètre selon les frameworks utilisés
C. Le stride est le pas de déplacement du filtre ; le padding ajoute des zéros en bordure ✓

> **Réponse : C** — Stride=2 divise la résolution par 2 ; padding='same' conserve la résolution de sortie égale à celle d'entrée.

---

**Q6.** À quoi sert le max pooling dans un CNN ?

A. Il normalise les activations pour éviter l'explosion des valeurs dans les couches profondes du réseau
B. Il réduit la résolution spatiale en conservant le maximum par région, apportant une invariance locale ✓
C. Il sélectionne les filtres les plus activés et supprime les autres pour régulariser et compresser le réseau

> **Réponse : B** — Le max pooling 2×2 divise la taille par 2 dans chaque dimension, réduisant le coût computationnel et la sensibilité aux translations exactes.

---

**Q7.** Pourquoi faire hériter son modèle de `nn.Module` dans PyTorch ?

A. `nn.Module` enregistre automatiquement les paramètres, gère le déplacement GPU et la sauvegarde du modèle ✓
B. C'est une convention de nommage qui améliore la lisibilité mais sans impact fonctionnel sur l'entraînement
C. `nn.Module` optimise automatiquement l'architecture du réseau en éliminant les couches redondantes

> **Réponse : A** — En déclarant les couches dans `__init__`, PyTorch sait quels tenseurs sont des paramètres apprenables. Cela débloque tout l'écosystème : passer les poids à l'optimiseur, les déplacer sur GPU, les sauvegarder et les recharger.

---

**Q8.** Comment les CNN apprennent-ils des représentations hiérarchiques des images ?

A. Les premières couches détectent des features simples ; les couches profondes, des structures complexes ✓
B. Chaque couche traite l'image entière à une résolution différente, comme une pyramide gaussienne multi-échelle
C. Les CNN utilisent une attention multi-têtes pour pondérer dynamiquement les différentes régions de l'image

> **Réponse : A** — Cette hiérarchie émergente est une propriété clé des CNN ; les features deviennent de plus en plus abstraites et sémantiques au fur et à mesure qu'on monte dans le réseau.

---

**Q9.** Quelles sont les étapes typiques d'une boucle d'entraînement PyTorch ?

A. `zero_grad → forward → compute loss → backward → optimizer.step` ✓
B. `forward → backward → compute loss → zero_grad → optimizer.step`
C. `zero_grad → forward → backward → optimizer.step → compute loss`

> **Réponse : A** — L'ordre est : remettre les gradients à zéro, calculer la prédiction, calculer la loss, propager les gradients, puis mettre à jour les poids.

---

## J4AM — Tokenisation & Embeddings

**Q1.** Quel est le principal inconvénient d'une tokenisation au niveau du mot (word-level) ?

A. Elle produit des séquences trop courtes et perd le contexte nécessaire au traitement du langage
B. Le vocabulaire devient énorme et les mots rares ou inconnus sont ignorés ✓
C. La tokenisation word-level est trop lente pour les corpus de grande taille utilisés en pré-entraînement

> **Réponse : B** — Un vocabulaire de 500k+ mots avec beaucoup de tokens rares est inefficace ; les mots hors-vocabulaire ("unknown") causent des pertes d'information.

---

**Q2.** Pourquoi la tokenisation subword (BPE) est-elle le compromis préféré aujourd'hui ?

A. BPE est plus rapide que les autres tokenisations car elle réduit le nombre de tokens par séquence
B. Elle gère les mots inconnus en les décomposant en sous-unités connues du vocabulaire ✓
C. BPE compresse les textes pour réduire la mémoire nécessaire à l'entraînement des modèles de langage

> **Réponse : B** — "unbelievable" → ["un", "believ", "able"] ; les sous-mots sont réutilisés entre mots apparentés, et aucun mot n'est vraiment inconnu.

---

**Q3.** Quel est le problème principal de l'encodage one-hot pour représenter des mots ?

A. Les vecteurs sont très creux et ne capturent aucune similarité sémantique entre les mots ✓
B. One-hot ne peut pas représenter plus de quelques milliers de mots sans dégrader les performances du modèle
C. L'encodage one-hot est trop lent à calculer et à stocker pour un vocabulaire de taille normale

> **Réponse : A** — "chat" et "chien" ont des vecteurs orthogonaux en one-hot (distance identique à "chat" et "aéroport"), alors qu'ils sont sémantiquement proches.

---

**Q4.** Qu'est-ce qu'un embedding dense ?

A. Une technique de compression qui réduit la taille des modèles de langage lors du déploiement en production
B. Un encodage binaire qui utilise tous les bits disponibles pour maximiser l'information par dimension
C. Une représentation vectorielle compacte apprise, où des mots similaires ont des vecteurs proches ✓

> **Réponse : C** — Contrairement au one-hot (creux, dimension = taille vocabulaire), un embedding dense est compact et encode la sémantique dans la géométrie de l'espace.

---

**Q5.** Qu'est-ce que l'hypothèse distributionnelle sur laquelle repose Word2Vec ?

A. Un mot est défini par ses contextes d'apparition : les mots similaires partagent des contextes similaires ✓
B. Les mots fréquents encodent plus d'information sémantique et doivent avoir des vecteurs de plus grande norme
C. La distribution des mots dans un corpus suit une loi de puissance qui contraint la forme des embeddings appris

> **Réponse : A** — « You shall know a word by the company it keeps » (Firth, 1957) ; Word2Vec exploite cette hypothèse en entraînant des vecteurs à prédire le contexte.

---

**Q6.** Que signifie que deux mots ont une similarité cosinus proche de 1 ?

A. Leurs vecteurs pointent dans la même direction, indiquant une forte similarité sémantique ou contextuelle ✓
B. Leurs vecteurs ont la même norme, ce qui signifie qu'ils apparaissent avec la même fréquence dans le corpus
C. Les deux mots sont interchangeables dans toutes les phrases du corpus d'entraînement sans changer le sens

> **Réponse : A** — La similarité cosinus mesure l'angle entre vecteurs, pas leur norme ; cos(θ)=1 → même direction → contextes d'utilisation similaires → sémantique proche.

---

**Q7.** Pourquoi des analogies comme "roi - homme + femme ≈ reine" émergent-elles des embeddings ?

A. Ces analogies sont programmées explicitement dans le dictionnaire de Word2Vec lors de son entraînement
B. C'est un artefact du preprocessing qui retire les stop words et crée ces relations de manière artificielle
C. L'entraînement fait émerger des directions vectorielles pour chaque relation sémantique régulière ✓

> **Réponse : C** — Ces structures algébriques émergent spontanément ; la direction "homme → femme" encode le genre et est cohérente pour de nombreuses paires.

---

**Q8.** Quel est le problème de la polysémie avec les embeddings statiques comme Word2Vec ?

A. Word2Vec ne peut pas représenter les mots polysémiques car ils nécessitent plusieurs entrées dans le vocabulaire
B. La polysémie multiplie le nombre de tokens et fait exploser la taille du vocabulaire lors de l'entraînement
C. Un mot polysémique reçoit un unique vecteur, moyenne floue de tous ses sens, sans distinction de contexte ✓

> **Réponse : C** — "banque" (établissement financier) et "banque" (rive d'une rivière) sont fusionnés en un seul vecteur qui ne capture bien aucun des deux sens.

---

**Q9.** Quelle information importante un sac de mots (bag-of-words) ignore-t-il ?

A. L'ordre des mots et les relations syntaxiques entre eux ✓
B. La fréquence d'apparition de chaque mot dans le document analysé
C. La longueur totale du document et le nombre de phrases qu'il contient

> **Réponse : A** — "le chat mange la souris" et "la souris mange le chat" ont le même bag-of-words ; l'ordre syntaxique, crucial pour le sens, est perdu.

---

## J4PM — Transformers & Fine-tuning

**Q1.** Quel avantage l'attention offre-t-elle par rapport aux RNNs pour traiter des séquences ?

A. Chaque position accède directement à toutes les autres en une seule opération, sans goulot séquentiel ✓
B. L'attention est plus simple à implémenter et consomme moins de mémoire GPU que les cellules récurrentes
C. L'attention est intrinsèquement bidirectionnelle alors que les RNNs ne traitent les séquences que dans un sens

> **Réponse : A** — Les RNNs souffrent du "long-range dependency" problem (l'information est diluée sur de longues séquences) ; l'attention y accède en O(1).

---

**Q2.** Dans le mécanisme d'attention, à quoi correspondent les Query, Key et Value ?

A. Query = données d'entrée brutes ; Key = poids du modèle ; Value = activation de la couche précédente
B. Query = ce qu'on cherche ; Key = ce avec quoi on compare ; Value = l'information récupérée ✓
C. Query, Key et Value sont trois projections linéaires sans interprétation sémantique propre

> **Réponse : B** — Le score d'attention = softmax(QKᵀ/√d) détermine combien "payer attention" à chaque Value ; c'est une recherche douce différentiable.

---

**Q3.** Quelle est la différence entre un Transformer encodeur (BERT) et décodeur (GPT) ?

A. L'encodeur lit les tokens en parallèle avec accès au contexte gauche et droit ; le décodeur masque les tokens futurs pour ne voir que le contexte passé lors de la génération
B. L'encodeur utilise des connexions résiduelles et de la normalisation ; le décodeur n'utilise que la normalisation sans connexion résiduelle
C. L'encodeur utilise une attention bidirectionnelle ; le décodeur génère token par token avec attention causale ✓

> **Réponse : C** — BERT est pré-entraîné avec masking (voit le contexte gauche ET droit) ; GPT prédit le token suivant (voit seulement le contexte gauche), d'où l'attention causale.

---

**Q4.** Pourquoi l'encodage positionnel est-il nécessaire dans un Transformer ?

A. L'attention est invariante à l'ordre ; sans encodage positionnel, le modèle ne distingue pas la position des mots ✓
B. L'encodage positionnel compresse les représentations intermédiaires pour réduire la mémoire consommée par le calcul d'attention en O(n²)
C. L'encodage positionnel permet de gérer des séquences de longueur variable sans recours à un padding fixe à l'entrée

> **Réponse : A** — Contrairement aux RNNs qui traitent séquentiellement, l'attention est une opération d'ensemble ; l'ordre doit être injecté explicitement via les encodages positionnels.

---

**Q5.** Qu'est-ce que le transfer learning ?

A. La réutilisation d'un modèle pré-entraîné comme initialisation pour une tâche spécifique ✓
B. La copie directe des poids d'un modèle source vers une architecture cible similaire, sans aucune phase de ré-entraînement ni adaptation
C. Le partage de jeux de données entre plusieurs équipes pour développer des modèles de manière collaborative et distribuée

> **Réponse : A** — Au lieu de partir de poids aléatoires, on initialise avec un modèle déjà entraîné ; les features apprises (bords pour CNN, syntaxe pour NLP) sont réutilisées.

---

**Q6.** Quelle est la différence entre pre-training et fine-tuning ?

A. Le pre-training est supervisé sur des données étiquetées spécifiques à la tâche ; le fine-tuning est non supervisé sur des données brutes génériques
B. Pre-training = entraînement général sur de grandes quantités de données ; fine-tuning = adaptation sur une tâche spécifique ✓
C. Pre-training et fine-tuning sont deux noms pour les deux phases d'un même entraînement continu sur le même dataset

> **Réponse : B** — BERT est pré-entraîné sur Wikipedia (~3B mots) puis fine-tuné sur quelques milliers d'exemples pour la classification de sentiment, par exemple.

---

**Q7.** Pourquoi utilise-t-on le F1-score plutôt que la précision (accuracy) pour évaluer un modèle de classification ?

A. Le F1-score est toujours supérieur à l'accuracy sur les datasets déséquilibrés, ce qui en fait la métrique de référence dans les benchmarks NLP et vision
B. Le F1-score est la seule métrique compatible avec les modèles Transformer et les tâches de génération de texte
C. L'accuracy est trompeuse sur des classes déséquilibrées : un score élevé peut masquer un modèle trivial ✓

> **Réponse : C** — Sur un dataset avec 95% de négatifs, un modèle qui prédit toujours "négatif" obtient 95% d'accuracy. Le F1-score combine précision et rappel, ce qui pénalise ce type de comportement.

---

**Q8.** Quel problème la RAG (Retrieval-Augmented Generation) résout-elle ?

A. Elle accélère l'inférence en récupérant des réponses pré-calculées depuis un cache vectoriel distribué
B. Elle compresse les LLMs pour les déployer sur des appareils à mémoire limitée sans perte de précision
C. Elle permet au LLM d'accéder à des informations non vues lors de l'entraînement ✓

> **Réponse : C** — Le LLM a des connaissances figées à la date de son entraînement ; RAG récupère des documents pertinents et les injecte dans le contexte pour ancrer la réponse.

---

**Q9.** Dans quel cas faut-il préférer le fine-tuning au prompt engineering ?

A. Quand on veut des réponses plus longues, plus formelles et plus détaillées de la part du modèle
B. Quand les données étiquetées disponibles permettent d'adapter le style ou les connaissances au-delà du prompting ✓
C. Le fine-tuning est toujours préférable car il internalise les connaissances plutôt que de les injecter au runtime

> **Réponse : B** — Le prompt engineering est moins coûteux et suffisant pour beaucoup de cas ; le fine-tuning vaut le coût quand la tâche est très spécifique ou que les performances du prompting sont insuffisantes.

---

## J5AM — MLOps & Industrialisation

**Q1.** Quel est le principal inconvénient de Pickle pour déployer un modèle en production ?

A. Un fichier Pickle est lié à une version Python et une version de bibliothèque — il peut échouer après une mise à jour ✓
B. Pickle est limité aux petits modèles et ne supporte pas les architectures profondes avec plus de 100 couches
C. Pickle compresse mal les poids du modèle et produit des fichiers plusieurs fois plus volumineux que les formats natifs

> **Réponse : A** — Un modèle sérialisé avec Pickle en Python 3.9 + PyTorch 1.x peut échouer à charger en Python 3.11 + PyTorch 2.x. C'est pour ça qu'on lui préfère ONNX en production.

---

**Q2.** Quel est l'avantage du format ONNX par rapport à un format natif PyTorch ?

A. ONNX produit des modèles plus petits et numériquement plus précis que les formats natifs PyTorch ou TensorFlow
B. ONNX est le seul format de modèle supporté nativement par les APIs cloud AWS, GCP et Azure
C. ONNX est interopérable : le modèle peut être exécuté par différents runtimes sans dépendance à PyTorch ✓

> **Réponse : C** — Entraîner en PyTorch et déployer via ONNX Runtime permet d'optimiser l'inférence indépendamment du framework d'entraînement.

---

**Q3.** Pourquoi faut-il éviter de charger le modèle dans la fonction de route d'une API ?

A. Les fonctions de route n'ont pas accès au système de fichiers pour lire un fichier modèle lors d'une requête HTTP
B. FastAPI impose que les modèles soient chargés dans un worker séparé pour éviter les conflits de threads
C. Le modèle serait rechargé à chaque requête, causant une latence et une consommation mémoire inacceptables ✓

> **Réponse : C** — Charger un modèle peut prendre plusieurs secondes ; le charger à l'initialisation de l'application garantit que toutes les requêtes utilisent la même instance déjà chargée.

---

**Q4.** Qu'est-ce que le "train/serve skew" ?

A. La différence de performance entre l'entraînement sur CPU et l'inférence sur GPU due aux précisions numériques
B. Les divergences entre les données d'entraînement et de production, causant une dégradation des performances ✓
C. Le décalage temporel entre le moment où un modèle est entraîné et celui où il est effectivement déployé

> **Réponse : B** — Ex : normalisation avec des statistiques différentes, features calculées différemment — le modèle voit des données en production différentes de ce qu'il a appris.

---

**Q5.** Qu'est-ce que la quantification d'un modèle ?

A. La mesure standardisée de la qualité d'un modèle avec des métriques de benchmarking sur des datasets publics
B. La compression du modèle en supprimant les connexions et neurones les moins actifs pendant l'entraînement
C. La réduction de la précision numérique des poids pour diminuer la taille du modèle et accélérer l'inférence ✓

> **Réponse : C** — Un modèle quantifié en int8 est ~4x plus petit et plus rapide à l'inférence ; c'est une technique clé pour le déploiement on-edge.

---

**Q6.** Quelle est la différence entre data drift et concept drift ?

A. Le data drift concerne les modèles NLP ; le concept drift concerne les modèles de vision par ordinateur
B. Data drift = la distribution des entrées change ; concept drift = la relation entre features et cible change ✓
C. Le data drift est détectable automatiquement par monitoring ; le concept drift nécessite une supervision humaine

> **Réponse : B** — Data drift : les mêmes features mais avec des valeurs différentes. Concept drift : "spam" signifie quelque chose de différent qu'à l'époque de l'entraînement.

---

**Q7.** Pourquoi containeriser un modèle avec Docker améliore-t-il la reproductibilité ?

A. Le container encapsule toutes les dépendances dans une image immuable qui s'exécute identiquement partout ✓
B. Docker chiffre les poids du modèle pour le protéger contre le vol de propriété intellectuelle en production
C. Docker orchestre automatiquement la distribution de l'inférence sur plusieurs machines via des replicas

> **Réponse : A** — "Works on my machine" disparaît : l'image Docker garantit que dev, staging et prod utilisent exactement le même environnement.

---

**Q8.** À quel problème répond un registre de modèles (model registry) ?

A. Il stocke les données d'entraînement pour permettre le ré-entraînement futur et la mise à jour des modèles
B. Il centralise le versioning des modèles, leurs métriques et leur cycle de vie pour faciliter les rollbacks ✓
C. Il remplace les tests unitaires en vérifiant automatiquement la qualité des modèles avant leur mise en production

> **Réponse : B** — Sans registre, savoir quel modèle tourne en prod, avec quelles données il a été entraîné, et comment revenir à une version précédente devient vite ingérable.

---

**Q9.** Quelle est la différence entre un déploiement canary et un déploiement A/B testing ?

A. Canary = rollout progressif avec rollback possible ; A/B = partage durable du trafic pour mesurer l'impact ✓
B. Canary = déploiement simultané sur tous les serveurs pour maximiser la couverture ; A/B = déploiement progressif serveur par serveur
C. Canary s'applique exclusivement aux modèles ML ; l'A/B testing est réservé aux applications web classiques

> **Réponse : A** — Canary vise à réduire le risque du déploiement ; A/B testing vise à mesurer l'impact métier — les objectifs et durées sont différents.

---

**Q10.** Pourquoi le CI/CD pour le ML diffère-t-il du CI/CD classique pour une application web ?

A. Le ML n'a pas besoin de tests automatisés car les modèles s'évaluent directement via leurs métriques de performance
B. Il faut aussi versionner données et modèles, et inclure des étapes d'entraînement et de validation des métriques ✓
C. Le CI/CD ML est plus simple car les modèles sont des boîtes noires sans tests unitaires ni d'intégration possibles

> **Réponse : B** — Un changement de données peut dégrader les performances sans toucher au code ; les tests de régression ML comparent les métriques entre versions du modèle.

---
