# QCM — IA, Deep Learning et Machine Learning

Questions orientées compréhension conceptuelle. Format : 3 options (A/B/C), une seule bonne réponse.

---

## J1AM — Fondations mathématiques & outils Python

**Q1.** Pourquoi préfère-t-on utiliser NumPy plutôt que des boucles Python pour les opérations sur les tableaux ?

A. NumPy est plus lisible et produit du code plus court
B. Python ne supporte pas les boucles sur des tableaux numériques
C. Les opérations vectorisées NumPy sont exécutées en C compilé, bien plus rapides que les boucles Python ✓

> **Réponse : C** — NumPy délègue les calculs à des routines C/Fortran optimisées ; les boucles Python sont interprétées et ont un overhead important par itération.

---

**Q2.** Qu'est-ce que le broadcasting en NumPy ?

A. La diffusion des calculs sur plusieurs cœurs CPU en parallèle
B. Un mécanisme qui permet d'opérer sur des tableaux de formes différentes en « étirant » implicitement les dimensions de taille 1 ✓
C. La copie d'un tableau en mémoire pour éviter les effets de bord

> **Réponse : B** — Le broadcasting évite de créer des copies inutiles : NumPy traite un vecteur `(n,)` comme s'il était répliqué pour s'aligner sur une matrice `(m, n)`.

---

**Q3.** Quelle est la différence entre `A @ B` et `A * B` pour deux matrices NumPy ?

A. `@` effectue le produit élément par élément, `*` effectue le produit matriciel
B. `@` effectue le produit matriciel (somme des produits ligne×colonne), `*` effectue le produit élément par élément ✓
C. Les deux sont équivalents si les matrices ont la même forme

> **Réponse : B** — `A @ B` est le produit matriciel (équivalent à `np.matmul`) : le résultat `(i,j)` est le produit scalaire de la ligne `i` de A avec la colonne `j` de B. `A * B` multiplie chaque élément avec son homologue — les formes doivent être compatibles au sens du broadcasting.

---

**Q4.** Quelle est la différence entre une matrice et un vecteur du point de vue d'une transformation linéaire ?

A. Un vecteur est une liste de scalaires, une matrice est juste plusieurs vecteurs empilés — il n'y a pas de différence conceptuelle
B. Une matrice encode une transformation linéaire (rotation, mise à l'échelle, projection) ; un vecteur est un point ou une direction dans l'espace ✓
C. Les matrices ne peuvent transformer que des vecteurs colonnes, jamais des vecteurs lignes

> **Réponse : B** — Multiplier `Ax` applique la transformation encodée par `A` au vecteur `x`, changeant potentiellement sa direction et sa norme.

---

**Q5.** Qu'est-ce qu'un gradient ?

A. La valeur de la fonction en un point donné
B. La dérivée seconde d'une fonction, indiquant sa courbure
C. Le vecteur des dérivées partielles d'une fonction scalaire, pointant dans la direction de plus forte croissance ✓

> **Réponse : C** — Pour descendre une fonction de perte, on se déplace dans la direction opposée au gradient (descente de gradient).

---

**Q6.** Pourquoi la règle de la chaîne (chain rule) est-elle centrale en deep learning ?

A. Elle permet de calculer la dérivée d'une fonction composée, ce qui est exactement ce que fait un réseau profond ✓
B. Elle simplifie les calculs matriciels en décomposant les produits
C. Elle garantit la convergence de la descente de gradient vers un minimum global

> **Réponse : A** — Un réseau de neurones est une composition de fonctions (couches) ; la rétropropagation applique la chain rule pour propager les gradients couche par couche.

---

**Q7.** Quelle est la différence entre un DataFrame Pandas et un tableau NumPy ?

A. Un DataFrame a des colonnes nommées, des index, et peut contenir des types hétérogènes — NumPy est homogène et sans étiquette ✓
B. Pandas est plus lent que NumPy car il est écrit en Python pur
C. NumPy peut stocker des données textuelles, pas Pandas

> **Réponse : A** — Pandas est conçu pour les données tabulaires avec métadonnées (noms de colonnes, index, types mixtes) ; NumPy est optimisé pour le calcul numérique homogène.

---

**Q8.** Pourquoi normaliser les features avant d'appliquer la descente de gradient ?

A. La normalisation est obligatoire pour que NumPy puisse effectuer les calculs matriciels
B. Sans normalisation, les features à grande échelle dominent le gradient — les pas sont trop grands dans certaines directions et trop petits dans d'autres ✓
C. La normalisation améliore uniquement la lisibilité du code, sans impact sur l'optimisation

> **Réponse : B** — Si une feature varie de 0 à 1000 et une autre de 0 à 1, le gradient sera très sensible à la première et quasi-insensible à la seconde. Normaliser rend les pas équilibrés dans toutes les directions et accélère la convergence.

---

## J1PM — Principes du ML & Régression linéaire

**Q1.** Quelle est la différence entre apprentissage supervisé, non supervisé et par renforcement ?

A. Supervisé = étiquettes fournies ; non supervisé = pas d'étiquettes ; renforcement = apprentissage par récompenses issues d'interactions avec un environnement ✓
B. Supervisé = données structurées ; non supervisé = données brutes ; renforcement = apprentissage en ligne
C. Ce sont trois noms différents pour la même chose selon le domaine d'application

> **Réponse : A** — La distinction clé est la nature du signal d'apprentissage : paires (x, y) étiquetées, structure des données seules, ou signal de récompense retardé.

---

**Q2.** Pourquoi ne peut-on pas évaluer un modèle uniquement sur ses données d'entraînement ?

A. Les données d'entraînement sont trop volumineuses pour être utilisées en évaluation
B. Les métriques calculées sur les données d'entraînement sont moins précises statistiquement
C. Un modèle peut apprendre par cœur les exemples d'entraînement (overfitting) sans avoir la capacité de généraliser ✓

> **Réponse : C** — L'objectif est la généralisation : un modèle qui mémorise son jeu d'entraînement obtient 100% mais échoue sur de nouvelles données.

---

**Q3.** Que représente la fonction de perte (loss function) dans un problème d'apprentissage ?

A. Le temps de calcul nécessaire pour entraîner le modèle
B. Le nombre de paramètres du modèle à optimiser
C. Une mesure de l'écart entre les prédictions du modèle et les valeurs réelles, que l'on cherche à minimiser ✓

> **Réponse : C** — La loss quantifie l'erreur ; l'entraînement consiste à ajuster les paramètres pour réduire cette erreur sur les données d'entraînement.

---

**Q4.** Quel est l'effet d'un taux d'apprentissage (learning rate) trop élevé lors de l'optimisation ?

A. Les mises à jour des poids sont trop grandes : on dépasse le minimum et la loss diverge ou oscille ✓
B. L'entraînement est plus lent car les mises à jour sont trop grandes
C. Le modèle converge plus vite vers un meilleur minimum global

> **Réponse : A** — Un learning rate trop grand fait « sauter » par-dessus le minimum ; trop petit, la convergence est lente — il faut trouver le bon équilibre.

---

**Q5.** Quelle est la différence entre le gradient descent en batch, stochastique et mini-batch ?

A. Batch = 1 exemple ; stochastique = tout le dataset ; mini-batch = un sous-ensemble
B. Ce sont des noms différents pour la même algorithme selon la taille du modèle
C. Batch = tout le dataset par itération ; stochastique = 1 exemple ; mini-batch = un sous-ensemble (ex: 32 exemples) ✓

> **Réponse : C** — La taille du lot utilisé pour calculer le gradient change le bruit des mises à jour et l'utilisation mémoire/GPU.

---

**Q6.** Pourquoi utilise-t-on le mini-batch en pratique plutôt que le batch ou le SGD pur ?

A. Le mini-batch est mathématiquement plus précis que les deux autres méthodes
B. Le mini-batch est la seule méthode compatible avec la rétropropagation
C. Il équilibre stabilité du gradient (vs SGD bruyant) et efficacité GPU (vectorisation) tout en permettant de traiter des datasets qui ne tiennent pas en RAM ✓

> **Réponse : C** — Le mini-batch exploite le parallélisme GPU (tenseurs en batch), introduit du bruit régularisant utile, et reste plus stable que le SGD pur.

---

**Q7.** Pourquoi ne peut-on pas toujours utiliser la solution analytique pour la régression linéaire ?

A. La solution analytique n'existe que pour les problèmes à une seule variable
B. La solution analytique donne des poids non interprétables
C. Inverser la matrice `(XᵀX)` a une complexité O(p³) et peut être instable si les features sont colinéaires, rendant la descente de gradient préférable pour les grands datasets ✓

> **Réponse : C** — L'équation normale `w = (XᵀX)⁻¹Xᵀy` est exacte mais coûteuse (p features) et numériquement instable ; le GD s'adapte mieux aux grandes dimensions.

---

**Q8.** Qu'est-ce que la convexité d'une fonction de perte nous garantit-elle pour l'optimisation ?

A. Que l'algorithme convergera en un nombre fixe d'itérations
B. Qu'il existe un unique minimum global, et que tout minimum local est aussi le minimum global ✓
C. Que le gradient est toujours positif, facilitant la descente

> **Réponse : B** — Pour une fonction convexe (comme la MSE de la régression linéaire), tout minimum local est global — la descente de gradient trouve la solution optimale.

---

## J2AM — Classification & Perceptron

**Q1.** Quelle est la différence fondamentale entre un problème de régression et un problème de classification ?

A. La régression utilise des réseaux de neurones, la classification utilise des arbres de décision
B. La régression prédit une valeur continue ; la classification prédit une catégorie (classe discrète) ✓
C. La régression fonctionne uniquement sur des données numériques, la classification sur des données textuelles

> **Réponse : B** — Ex : prédire un prix (régression) vs prédire si un email est spam ou non (classification).

---

**Q2.** Pourquoi la fonction `sign` ne peut-elle pas être utilisée directement dans une descente de gradient ?

A. La fonction sign est trop lente à calculer pour les grands réseaux
B. Elle n'est pas différentiable : son gradient est nul partout sauf en 0 où il est indéfini, rendant la rétropropagation impossible ✓
C. La fonction sign produit des valeurs trop grandes qui déstabilisent l'entraînement

> **Réponse : B** — La descente de gradient requiert des gradients utilisables ; `sign` a un gradient nul, ce qui bloque toute mise à jour des poids.

---

**Q3.** Quel est le rôle d'une fonction d'activation non-linéaire dans un réseau de neurones ?

A. Elle normalise les activations pour éviter l'explosion des gradients
B. Elle permet au réseau d'apprendre des frontières de décision non linéaires ; sans elle, empiler des couches linéaires reste linéaire ✓
C. Elle convertit les sorties en probabilités entre 0 et 1

> **Réponse : B** — Sans non-linéarité, `W₂(W₁x) = (W₂W₁)x` : le réseau entier se réduit à une seule transformation linéaire.

---

**Q4.** Pourquoi ReLU est-elle préférée à d'autres fonctions non-linéaires en pratique ?

A. Son gradient est constant (1 pour x>0), ce qui évite le problème du gradient qui disparaît (vanishing gradient) et son calcul est très rapide ✓
B. ReLU est la seule fonction d'activation qui garantit la convergence du réseau
C. ReLU produit toujours des sorties entre -1 et 1, stabilisant l'entraînement

> **Réponse : A** — Sigmoid et tanh saturent (gradient → 0), bloquant la rétropropagation dans les couches profondes. ReLU n'a pas ce problème pour x>0.

---

**Q5.** Quelle est la limitation fondamentale du perceptron de Rosenblatt à une seule couche ?

A. Il ne peut apprendre que des frontières de décision linéaires, et est incapable de séparer des données non linéairement séparables ✓
B. Il est trop lent à entraîner pour des datasets de grande taille
C. Il ne fonctionne que pour la classification binaire, pas multi-classe

> **Réponse : A** — Le théorème de convergence du perceptron garantit qu'il trouve une solution si et seulement si les données sont linéairement séparables.

---

**Q6.** Pourquoi le problème XOR est-il un cas emblématique en deep learning ?

A. XOR est le problème le plus difficile en optimisation combinatoire
B. Il démontre qu'un perceptron simple ne peut pas apprendre XOR (non linéairement séparable), mais qu'un MLP avec couche cachée le résout ✓
C. XOR est utilisé pour initialiser les poids des réseaux profonds

> **Réponse : B** — Minsky & Papert (1969) ont utilisé XOR pour montrer les limites du perceptron ; l'ajout d'une couche cachée suffit à le résoudre.

---

**Q7.** À quoi sert la fonction softmax dans une classification multi-classe ?

A. Elle convertit un vecteur de scores bruts (logits) en une distribution de probabilités qui somme à 1 ✓
B. Elle sélectionne la classe avec le score le plus élevé parmi toutes les classes
C. Elle normalise les entrées du réseau pour accélérer l'entraînement

> **Réponse : A** — Softmax(`zᵢ`) = `exp(zᵢ) / Σ exp(zⱼ)` ; chaque sortie est entre 0 et 1 et la somme vaut 1 — interprétable comme une probabilité.

---

**Q8.** Que signifie que la sortie du softmax est une distribution de probabilité ?

A. Les sorties sont triées par ordre décroissant de probabilité
B. Le réseau garantit que sa prédiction est correcte avec la probabilité indiquée
C. Chaque valeur est entre 0 et 1, et leur somme vaut exactement 1 — on peut interpréter chaque sortie comme la probabilité d'appartenir à une classe ✓

> **Réponse : C** — C'est une propriété mathématique du softmax, pas une garantie de calibration ; un modèle peut être confiant et faux.

---

**Q9.** Pourquoi ajouter des couches cachées avec des non-linéarités permet-il de résoudre XOR ?

A. Chaque couche apprend une représentation intermédiaire ; la couche cachée peut transformer l'espace des features en un espace linéairement séparable ✓
B. Les couches supplémentaires augmentent le nombre de paramètres, permettant la mémorisation
C. Les non-linéarités introduisent du bruit qui aide à éviter l'overfitting sur XOR

> **Réponse : A** — La couche cachée réorganise l'espace d'entrée ; dans la nouvelle représentation, XOR devient linéairement séparable.

---

## J2PM — Validation & Généralisation

**Q1.** Comment distingue-t-on l'overfitting de l'underfitting en pratique ?

A. Overfitting = loss d'entraînement élevée ; underfitting = loss d'entraînement basse
B. Overfitting = modèle trop petit ; underfitting = modèle trop grand
C. Overfitting = faible loss d'entraînement mais forte loss de validation ; underfitting = forte loss sur les deux ✓

> **Réponse : C** — L'overfitting se détecte par l'écart entre les courbes train et validation ; l'underfitting se voit par de mauvaises performances partout.

---

**Q2.** Pourquoi un modèle avec 100% de précision en entraînement n'est-il pas forcément bon ?

A. 100% de précision indique un bug dans le code d'entraînement
B. La précision n'est pas une métrique fiable ; il faut toujours utiliser la F1-score
C. Le modèle a peut-être mémorisé les données d'entraînement (overfitting) sans apprendre les patterns généralisables ✓

> **Réponse : C** — Un modèle qui mémorise les exemples d'entraînement atteint 100%, mais ses performances sur de nouvelles données peuvent être médiocres.

---

**Q3.** À quoi sert le jeu de validation, distinct du jeu de test ?

A. Le jeu de validation est utilisé pour l'entraînement final ; le jeu de test pour les itérations intermédiaires
B. Le jeu de validation est plus petit que le jeu de test pour économiser des données d'entraînement
C. Le jeu de validation sert à sélectionner les hyperparamètres et l'architecture sans « contaminer » le jeu de test, qui ne sert qu'à l'évaluation finale ✓

> **Réponse : C** — Utiliser le test set pour choisir un modèle revient à optimiser dessus ; le test set doit rester « invisible » jusqu'à l'évaluation finale.

---

**Q4.** Qu'est-ce que le compromis biais-variance ?

A. Le choix entre un modèle rapide (biais élevé) et un modèle précis (variance élevée)
B. Le biais mesure l'erreur sur les données d'entraînement, la variance mesure l'erreur sur le test
C. Un modèle complexe a peu de biais mais forte variance (sensible aux données) ; un modèle simple a fort biais mais faible variance ; l'erreur totale est la somme des deux ✓

> **Réponse : C** — Erreur = Biais² + Variance + Bruit irréductible ; trouver la bonne complexité du modèle minimise cette somme.

---

**Q5.** Un modèle trop simple souffre-t-il plutôt de biais ou de variance ?

A. De variance élevée, car il ne peut pas s'adapter aux données
B. Ni l'un ni l'autre — un modèle simple est toujours préférable (principe du rasoir d'Occam)
C. De biais élevé, car il fait des hypothèses trop restrictives sur la forme de la relation entre features et cible ✓

> **Réponse : C** — Un modèle trop simple (ex : régression linéaire sur des données non linéaires) sous-ajuste systématiquement : c'est du biais (erreur structurelle).

---

**Q6.** Pourquoi utilise-t-on la validation croisée (k-fold) plutôt qu'un simple split train/test ?

A. La k-fold cross-validation est plus rapide à calculer qu'un simple split
B. Elle fournit une estimation plus robuste des performances en utilisant toutes les données comme validation à tour de rôle, réduisant la variance de l'estimation ✓
C. La k-fold garantit que le modèle ne fait jamais d'overfitting

> **Réponse : B** — Avec un seul split, l'estimation dépend du hasard du split ; la k-fold moyenne sur k splits, donnant une estimation plus fiable et un intervalle de confiance.

---

**Q7.** Quelle est la différence entre un hyperparamètre et un paramètre appris ?

A. Les hyperparamètres sont des paramètres internes du modèle ; les paramètres appris sont fournis par l'utilisateur
B. Il n'y a pas de différence : les deux sont optimisés pendant l'entraînement
C. Les paramètres appris (poids, biais) sont optimisés par la descente de gradient ; les hyperparamètres (learning rate, nb de couches) sont fixés avant l'entraînement ✓

> **Réponse : C** — Les hyperparamètres contrôlent le processus d'apprentissage et ne sont pas mis à jour par rétropropagation ; on les choisit par validation croisée ou grid search.

---

**Q8.** Pourquoi ne doit-on jamais utiliser le jeu de test pour sélectionner un modèle ?

A. Le jeu de test est trop petit pour fournir des métriques significatives
B. Utiliser le test set pour choisir revient à s'y adapter (data leakage), rendant l'évaluation finale optimiste et non représentative des vraies performances en production ✓
C. Les métriques sur le test set sont moins fiables que celles sur le validation set

> **Réponse : B** — Si on choisit le meilleur modèle selon le test set, on optimise indirectement dessus ; l'estimation des performances en production sera biaisée.

---

## J3AM — Autograd & Rétropropagation

**Q1.** Pourquoi a-t-on besoin de la rétropropagation pour entraîner un réseau profond ?

A. Elle calcule efficacement le gradient de la loss par rapport à chaque paramètre du réseau en appliquant la chain rule de la sortie vers l'entrée ✓
B. La rétropropagation accélère le calcul des prédictions (forward pass)
C. Elle permet de paralléliser l'entraînement sur plusieurs GPUs

> **Réponse : A** — Sans rétropropagation, calculer ∂L/∂wᵢ pour chaque poids serait prohibitivement coûteux ; backprop le fait en un seul passage arrière.

---

**Q2.** Qu'est-ce que le "problème de l'attribution du crédit" (credit assignment problem) ?

A. La difficulté à déterminer quelle(s) connexion(s) ou neurone(s) du réseau sont responsables d'une erreur de prédiction, surtout dans les couches profondes ✓
B. Le problème de distribuer les données d'entraînement équitablement entre les couches
C. La gestion des conflits quand plusieurs gradients s'appliquent au même poids

> **Réponse : A** — Chaque couche contribue à l'erreur finale ; backprop résout ce problème en propageant les gradients couche par couche depuis la sortie.

---

**Q3.** Qu'est-ce qu'un graphe de calcul (computation graph) ?

A. Un graphique montrant la progression de la loss au fil des epochs
B. Une représentation sous forme de graphe acyclique dirigé (DAG) des opérations mathématiques d'un modèle, où les nœuds sont les valeurs et les arêtes les opérations ✓
C. Un diagramme de l'architecture du réseau de neurones (couches et connexions)

> **Réponse : B** — Le graphe de calcul trace toutes les opérations effectuées lors du forward pass, ce qui permet de calculer automatiquement les gradients en le parcourant à l'envers.

---

**Q4.** Quelle est la différence entre la passe avant (forward pass) et la passe arrière (backward pass) ?

A. Forward = entraînement ; backward = inférence
B. Forward = calcul de la prédiction et de la loss en propageant les données de l'entrée vers la sortie ; backward = calcul des gradients en propageant l'erreur de la sortie vers l'entrée ✓
C. Forward = calcul des gradients ; backward = mise à jour des poids

> **Réponse : B** — Ces deux passes constituent une itération d'entraînement : forward pour calculer la loss, backward pour calculer les gradients, puis mise à jour des paramètres.

---

**Q5.** Pourquoi empiler des couches linéaires sans non-linéarités entre elles n'a aucun intérêt ?

A. La composition de transformations linéaires est elle-même une transformation linéaire : `W₂(W₁x) = (W₂W₁)x`, donc tout le réseau reste équivalent à une seule couche ✓
B. Les couches linéaires sont trop lentes à calculer en séquence
C. Les gradients explosent lors de la rétropropagation à travers plusieurs couches linéaires

> **Réponse : A** — Peu importe le nombre de couches linéaires empilées, le modèle reste une régression linéaire ; les non-linéarités sont indispensables pour l'expressivité.

---

**Q6.** Comment les gradients se propagent-ils lorsqu'un nœud a plusieurs chemins entrants ?

A. Seul le chemin avec le gradient le plus élevé est conservé (max-gradient)
B. Les gradients des différents chemins s'additionnent au nœud (règle de somme des gradients) ✓
C. Les gradients des différents chemins sont multipliés entre eux

> **Réponse : B** — Dans backprop, quand un nœud a plusieurs sorties (ou reçoit des gradients depuis plusieurs chemins), on additionne les contributions — c'est la règle des dérivées partielles.

---

**Q7.** Pourquoi faut-il remettre les gradients à zéro (`zero_grad`) avant chaque itération d'entraînement ?

A. Pour libérer la mémoire GPU après chaque batch
B. PyTorch accumule les gradients par défaut ; sans remise à zéro, les gradients du batch courant s'ajoutent aux précédents, faussant les mises à jour ✓
C. Pour empêcher les gradients de dépasser 1 et déstabiliser l'entraînement

> **Réponse : B** — Ce comportement d'accumulation est intentionnel (utile pour les gradients accumulés sur plusieurs batches), mais doit être géré explicitement à chaque itération normale.

---

## J3PM — PyTorch & CNN

**Q1.** Quelle est la différence entre un tenseur PyTorch et un tableau NumPy ?

A. Les tenseurs PyTorch peuvent résider sur GPU et participent au graphe de calcul pour la différentiation automatique ; les tableaux NumPy sont CPU uniquement et sans autograd ✓
B. Les tenseurs PyTorch ne supportent que les entiers ; NumPy supporte les flottants
C. NumPy est plus rapide que PyTorch pour toutes les opérations numériques

> **Réponse : A** — PyTorch étend NumPy avec le support GPU et l'autograd, essentiels pour entraîner des réseaux de neurones efficacement.

---

**Q2.** À quoi sert le flag `requires_grad` dans PyTorch ?

A. Il indique que le tenseur doit être copié avant toute opération pour éviter les modifications en place
B. Il force le tenseur à rester sur CPU même si un GPU est disponible
C. Il signale à PyTorch de suivre les opérations sur ce tenseur dans le graphe de calcul, permettant de calculer automatiquement son gradient lors du backward ✓

> **Réponse : C** — Seuls les tenseurs avec `requires_grad=True` (les poids du modèle) accumulent les gradients ; les données d'entrée n'en ont pas besoin.

---

**Q3.** Pourquoi un MLP (réseau dense) est-il peu adapté au traitement d'images ?

A. Les MLP ne supportent pas les entrées en 2D ; il faut d'abord les convertir en 1D
B. Chaque pixel est connecté à tous les neurones : nombre de paramètres explosif, pas d'invariance aux translations, et les features spatiales locales sont ignorées ✓
C. Les MLP produisent des sorties continues, inadaptées à la classification d'images

> **Réponse : B** — Une image 224×224×3 donne ~150k entrées ; avec un MLP, la première couche seule aurait des millions de paramètres, sans exploiter la structure spatiale.

---

**Q4.** Qu'est-ce que le partage de poids dans une convolution ?

A. Le même filtre (noyau) est appliqué à toutes les positions de l'image, réduisant drastiquement le nombre de paramètres et apprenant des features invariantes aux translations ✓
B. Les poids sont partagés entre plusieurs réseaux entraînés en parallèle (data parallelism)
C. Les couches adjacentes d'un CNN partagent la moitié de leurs poids pour accélérer l'entraînement

> **Réponse : A** — Un filtre de taille 3×3×3 a seulement 27 paramètres quelle que soit la taille de l'image ; il détecte le même pattern où qu'il apparaisse.

---

**Q5.** Quelle est la différence entre stride et padding dans une couche de convolution ?

A. Le stride contrôle la taille du filtre ; le padding contrôle le nombre de filtres
B. Stride et padding sont deux noms pour le même paramètre selon les frameworks
C. Le stride est le pas de déplacement du filtre (contrôle la réduction spatiale) ; le padding ajoute des zéros en bordure pour contrôler la taille de sortie ✓

> **Réponse : C** — Stride=2 divise la résolution par 2 ; padding='same' conserve la résolution de sortie égale à celle d'entrée.

---

**Q6.** À quoi sert le max pooling dans un CNN ?

A. Il normalise les activations pour éviter l'explosion des valeurs dans les couches profondes
B. Il réduit la résolution spatiale en conservant la valeur maximale par région, diminuant le nombre de paramètres et apportant une invariance locale aux petits décalages ✓
C. Il sélectionne les filtres les plus importants et supprime les autres pour régulariser le réseau

> **Réponse : B** — Le max pooling 2×2 divise la taille par 2 dans chaque dimension, réduisant le coût computationnel et la sensibilité aux translations exactes.

---

**Q7.** Pourquoi faire hériter son modèle de `nn.Module` dans PyTorch ?

A. `nn.Module` enregistre automatiquement les paramètres et offre des fonctionnalités essentielles ✓
B. C'est une convention de nommage sans impact fonctionnel
C. `nn.Module` optimise automatiquement l'architecture du réseau selon les données

> **Réponse : A** — En déclarant les couches dans `__init__`, PyTorch sait quels tenseurs sont des paramètres apprenables. Cela débloque tout l'écosystème : passer les poids à l'optimiseur, les déplacer sur GPU, les sauvegarder et les recharger.

---

**Q8.** Comment les CNN apprennent-ils des représentations hiérarchiques des images ?

A. Les premières couches détectent des features simples (bords, couleurs) ; les couches profondes combinent ces features pour détecter des structures complexes (yeux, visages, objets) ✓
B. Chaque couche voit l'image entière à une résolution différente (pyramide gaussienne)
C. Les CNN utilisent une attention multi-têtes pour pondérer les différentes régions de l'image

> **Réponse : A** — Cette hiérarchie émergente est une propriété clé des CNN ; les features deviennent de plus en plus abstraites et sémantiques au fur et à mesure qu'on monte dans le réseau.

---

**Q9.** Quelles sont les étapes typiques d'une boucle d'entraînement PyTorch ?

A. `zero_grad → forward → compute loss → backward → optimizer.step` ✓
B. `forward → backward → zero_grad → optimizer.step`
C. `optimizer.step → forward → compute loss → backward → zero_grad`

> **Réponse : A** — L'ordre est : remettre les gradients à zéro, calculer la prédiction, calculer la loss, propager les gradients, puis mettre à jour les poids.

---

## J4AM — Tokenisation & Embeddings

**Q1.** Quel est le principal inconvénient d'une tokenisation au niveau du mot (word-level) ?

A. Elle produit des séquences trop courtes, perdant le contexte nécessaire au modèle
B. Le vocabulaire devient énorme (tous les mots possibles), les mots rares/inconnus sont ignorés (OOV), et les variantes morphologiques sont traitées comme des tokens distincts ✓
C. La tokenisation word-level est trop lente pour les modèles de langage modernes

> **Réponse : B** — Un vocabulaire de 500k+ mots avec beaucoup de tokens rares est inefficace ; les mots hors-vocabulaire ("unknown") causent des pertes d'information.

---

**Q2.** Pourquoi la tokenisation subword (BPE) est-elle le compromis préféré aujourd'hui ?

A. BPE est plus rapide que la tokenisation word-level ou character-level
B. Elle gère les mots inconnus en les décomposant en sous-unités connues, réduit la taille du vocabulaire, et partage les représentations de racines entre mots morphologiquement liés ✓
C. BPE compresse les textes pour réduire la mémoire nécessaire à l'entraînement

> **Réponse : B** — "unbelievable" → ["un", "believ", "able"] ; les sous-mots sont réutilisés entre mots apparentés, et aucun mot n'est vraiment inconnu.

---

**Q3.** Quel est le problème principal de l'encodage one-hot pour représenter des mots ?

A. Les vecteurs sont de dimension égale à la taille du vocabulaire (très grands et creux) et ne capturent aucune similarité sémantique entre les mots ✓
B. One-hot ne peut pas représenter plus de 1000 mots différents
C. L'encodage one-hot est trop lent à calculer pour un vocabulaire de taille normale

> **Réponse : A** — "chat" et "chien" ont des vecteurs orthogonaux en one-hot (distance identique à "chat" et "aéroport"), alors qu'ils sont sémantiquement proches.

---

**Q4.** Qu'est-ce qu'un embedding dense ?

A. Une technique de compression qui réduit la taille des modèles de langage
B. Un encodage qui utilise tous les bits disponibles pour maximiser l'information par dimension
C. Une représentation vectorielle de dimension réduite (ex: 300 dimensions) apprise par le réseau, où des mots sémantiquement similaires ont des vecteurs proches ✓

> **Réponse : C** — Contrairement au one-hot (creux, dimension = taille vocabulaire), un embedding dense est compact et encode la sémantique dans la géométrie de l'espace.

---

**Q5.** Qu'est-ce que l'hypothèse distributionnelle sur laquelle repose Word2Vec ?

A. La signification d'un mot peut être inférée de son contexte : les mots qui apparaissent dans des contextes similaires ont des significations similaires ✓
B. Les mots fréquents sont plus importants que les mots rares pour la signification d'un texte
C. La distribution statistique des mots dans un corpus suit une loi de puissance (loi de Zipf)

> **Réponse : A** — « You shall know a word by the company it keeps » (Firth, 1957) ; Word2Vec exploite cette hypothèse en entraînant des vecteurs à prédire le contexte.

---

**Q6.** Que signifie que deux mots ont une similarité cosinus proche de 1 ?

A. Leurs vecteurs pointent dans la même direction dans l'espace des embeddings, indiquant une forte similarité sémantique ou contextuelle ✓
B. Leurs vecteurs d'embedding sont de même norme (même fréquence dans le corpus)
C. Les deux mots apparaissent exactement le même nombre de fois dans le corpus d'entraînement

> **Réponse : A** — La similarité cosinus mesure l'angle entre vecteurs, pas leur norme ; cos(θ)=1 → même direction → contextes d'utilisation similaires → sémantique proche.

---

**Q7.** Pourquoi des analogies comme "roi - homme + femme ≈ reine" émergent-elles des embeddings ?

A. Ces analogies sont programmées explicitement dans le dictionnaire de Word2Vec
B. C'est un artefact du preprocessing qui retire les stop words et crée ces relations artificiellement
C. L'entraînement sur de grands corpus fait émerger des directions vectorielles correspondant à des relations sémantiques (genre, royauté) régulières dans le langage ✓

> **Réponse : C** — Ces structures algébriques émergent spontanément ; la direction "homme → femme" encode le genre et est cohérente pour de nombreuses paires.

---

**Q8.** Quel est le problème de la polysémie avec les embeddings statiques comme Word2Vec ?

A. Word2Vec ne peut pas représenter les mots polysémiques car ils ont plusieurs orthographes
B. La polysémie fait exploser la taille du vocabulaire, rendant le modèle trop lent
C. Un mot polysémique (ex: "banque") reçoit un unique vecteur, moyenne floue de tous ses sens — les modèles contextuels (BERT) résolvent ce problème avec des représentations dépendant du contexte ✓

> **Réponse : C** — "banque" (établissement financier) et "banque" (rive d'une rivière) sont fusionnés en un seul vecteur qui ne capture bien aucun des deux sens.

---

**Q9.** Quelle information importante un sac de mots (bag-of-words) ignore-t-il ?

A. L'ordre des mots et les relations syntaxiques entre eux ✓
B. La fréquence des mots dans le document
C. La longueur totale du document

> **Réponse : A** — "le chat mange la souris" et "la souris mange le chat" ont le même bag-of-words ; l'ordre syntaxique, crucial pour le sens, est perdu.

---

## J4PM — Transformers & Fine-tuning

**Q1.** Quel avantage l'attention offre-t-elle par rapport aux RNNs pour traiter des séquences ?

A. L'attention permet à chaque position de la séquence d'accéder directement à toutes les autres positions en une seule opération, éliminant le goulot d'étranglement séquentiel des RNNs ✓
B. L'attention est plus simple à implémenter et nécessite moins de mémoire GPU
C. L'attention est bidirectionnelle alors que les RNNs ne peuvent traiter les séquences que dans un sens

> **Réponse : A** — Les RNNs souffrent du "long-range dependency" problem (l'information est diluée sur de longues séquences) ; l'attention y accède en O(1).

---

**Q2.** Dans le mécanisme d'attention, à quoi correspondent les Query, Key et Value ?

A. Query = données d'entrée brutes ; Key = poids du modèle ; Value = prédiction finale
B. Query = ce qu'on cherche ; Key = ce avec quoi on compare (index) ; Value = l'information récupérée si la clé correspond — analogie avec une base de données différentiable ✓
C. Query, Key et Value sont trois couches linéaires indépendantes sans interprétation sémantique particulière

> **Réponse : B** — Le score d'attention = softmax(QKᵀ/√d) détermine combien "payer attention" à chaque Value ; c'est une recherche douce différentiable.

---

**Q3.** Quelle est la différence entre un Transformer encodeur (BERT) et décodeur (GPT) ?

A. L'encodeur traite des images, le décodeur traite du texte
B. L'encodeur utilise des connexions résiduelles, le décodeur non
C. L'encodeur lit toute la séquence en parallèle avec attention bidirectionnelle (adapté à la compréhension) ; le décodeur génère token par token avec attention causale (adapté à la génération) ✓

> **Réponse : C** — BERT est pré-entraîné avec masking (voit le contexte gauche ET droit) ; GPT prédit le token suivant (voit seulement le contexte gauche), d'où l'attention causale.

---

**Q4.** Pourquoi l'encodage positionnel est-il nécessaire dans un Transformer ?

A. L'attention est invariante à l'ordre des tokens ; sans encodage positionnel, le modèle ne saurait pas distinguer "chien mord homme" de "homme mord chien" ✓
B. Il compresse les séquences longues pour réduire la mémoire nécessaire à l'attention
C. Il permet au modèle de traiter des séquences de longueur variable sans padding

> **Réponse : A** — Contrairement aux RNNs qui traitent séquentiellement, l'attention est une opération d'ensemble ; l'ordre doit être injecté explicitement via les encodages positionnels.

---

**Q5.** Qu'est-ce que le transfer learning ?

A. La réutilisation d'un modèle pré-entraîné sur une grande tâche générale comme point de départ pour une tâche spécifique, transférant les représentations apprises ✓
B. La copie des poids d'un modèle vers un autre sans ré-entraînement
C. Le transfert de données d'entraînement entre différentes équipes pour entraîner des modèles partagés

> **Réponse : A** — Au lieu de partir de poids aléatoires, on initialise avec un modèle déjà entraîné ; les features apprises (bords pour CNN, syntaxe pour NLP) sont réutilisées.

---

**Q6.** Quelle est la différence entre pre-training et fine-tuning ?

A. Pre-training est supervisé, fine-tuning est non supervisé
B. Pre-training = entraînement sur une grande tâche générale (ex: prédiction de mots masqués) avec beaucoup de données ; fine-tuning = adaptation sur une tâche spécifique avec peu de données ✓
C. Pre-training et fine-tuning désignent les deux phases d'un même entraînement continu

> **Réponse : B** — BERT est pré-entraîné sur Wikipedia (~3B mots) puis fine-tuné sur quelques milliers d'exemples pour la classification de sentiment, par exemple.

---

**Q7.** Pourquoi utilise-t-on le F1-score plutôt que la précision (accuracy) pour évaluer un modèle de classification ?

A. Le F1-score est toujours plus élevé que l'accuracy, ce qui le rend plus optimiste
B. Le F1-score est la seule métrique compatible avec les modèles de type Transformer
C. L'accuracy est trompeuse sur des données déséquilibrées — un modèle qui prédit toujours la classe majoritaire obtient un score élevé sans rien apprendre ✓

> **Réponse : C** — Sur un dataset avec 95% de négatifs, un modèle qui prédit toujours "négatif" obtient 95% d'accuracy. Le F1-score combine précision et rappel, ce qui pénalise ce type de comportement.

---

**Q8.** Quel problème la RAG (Retrieval-Augmented Generation) résout-elle ?

A. Elle accélère l'inférence en récupérant des réponses pré-calculées dans un cache
B. Elle compresse les LLMs pour les déployer sur des appareils avec peu de mémoire
C. Elle permet au LLM d'accéder à des informations récentes ou spécifiques au domaine non vues à l'entraînement, réduisant les hallucinations ✓

> **Réponse : C** — Le LLM a des connaissances figées à la date de son entraînement ; RAG récupère des documents pertinents et les injecte dans le contexte pour ancrer la réponse.

---

**Q9.** Dans quel cas faut-il préférer le fine-tuning au prompt engineering ?

A. Quand on veut des réponses plus longues et détaillées du modèle
B. Quand on a suffisamment de données étiquetées pour une tâche spécifique et qu'on a besoin d'adapter le style, le format ou les connaissances du modèle au-delà de ce qu'un prompt peut obtenir ✓
C. Le fine-tuning est toujours préférable au prompt engineering car il donne de meilleures performances

> **Réponse : B** — Le prompt engineering est moins coûteux et suffisant pour beaucoup de cas ; le fine-tuning vaut le coût quand la tâche est très spécifique ou que les performances du prompting sont insuffisantes.

---

## J5AM — MLOps & Industrialisation

**Q1.** Quel est le principal inconvénient de Pickle pour déployer un modèle en production ?

A. Un fichier Pickle est lié à une version Python et une version de bibliothèque spécifiques — il peut ne plus fonctionner après une mise à jour ✓
B. Pickle est limité aux petits modèles et ne supporte pas les réseaux de neurones profonds
C. Pickle compresse mal les modèles, produisant des fichiers trop volumineux

> **Réponse : A** — Un modèle sérialisé avec Pickle en Python 3.9 + PyTorch 1.x peut échouer à charger en Python 3.11 + PyTorch 2.x. C'est pour ça qu'on lui préfère ONNX en production.

---

**Q2.** Quel est l'avantage du format ONNX par rapport à un format natif PyTorch ?

A. ONNX produit des modèles plus petits et plus précis que les formats natifs
B. ONNX est le seul format supporté par les APIs cloud (AWS, GCP, Azure)
C. ONNX est un format interopérable : le modèle peut être exécuté par différents runtimes (ONNX Runtime, TensorRT, OpenVINO) et sur différentes plateformes sans dépendance à PyTorch ✓

> **Réponse : C** — Entraîner en PyTorch et déployer via ONNX Runtime permet d'optimiser l'inférence indépendamment du framework d'entraînement.

---

**Q3.** Pourquoi faut-il éviter de charger le modèle dans la fonction de route d'une API ?

A. Les fonctions de route n'ont pas accès au système de fichiers pour lire le modèle
B. FastAPI n'est pas compatible avec les modèles PyTorch ou TensorFlow
C. Le modèle serait rechargé à chaque requête, causant une latence inacceptable et une consommation mémoire excessive — il doit être chargé une fois au démarrage (pattern Singleton) ✓

> **Réponse : C** — Charger un modèle peut prendre plusieurs secondes ; le charger à l'initialisation de l'application garantit que toutes les requêtes utilisent la même instance déjà chargée.

---

**Q4.** Qu'est-ce que le "train/serve skew" ?

A. La différence de performance entre l'entraînement sur CPU et l'inférence sur GPU
B. Les divergences entre les données/preprocessing d'entraînement et ceux utilisés en production, causant une dégradation des performances malgré de bonnes métriques de validation ✓
C. Le décalage temporel entre le moment où un modèle est entraîné et celui où il est déployé

> **Réponse : B** — Ex : normalisation avec des statistiques différentes, features calculées différemment — le modèle voit des données en production différentes de ce qu'il a appris.

---

**Q5.** Qu'est-ce que la quantification d'un modèle ?

A. La mesure de la qualité d'un modèle avec des métriques standardisées
B. La compression du modèle en supprimant les couches inutilisées (pruning)
C. La réduction de la précision numérique des poids (ex: float32 → int8) pour réduire la taille du modèle et accélérer l'inférence, au prix d'une légère perte de précision ✓

> **Réponse : C** — Un modèle quantifié en int8 est ~4x plus petit et plus rapide à l'inférence ; c'est une technique clé pour le déploiement on-edge.

---

**Q6.** Quelle est la différence entre data drift et concept drift ?

A. Data drift concerne les modèles de NLP, concept drift les modèles de vision
B. Data drift = la distribution des données d'entrée change (ex: nouveaux utilisateurs) ; concept drift = la relation entre les features et la cible change (ex: le comportement des utilisateurs évolue) ✓
C. Data drift est détectable automatiquement, concept drift nécessite une supervision humaine

> **Réponse : B** — Data drift : les mêmes features mais avec des valeurs différentes. Concept drift : "spam" signifie quelque chose de différent qu'à l'époque de l'entraînement.

---

**Q7.** Pourquoi containeriser un modèle avec Docker améliore-t-il la reproductibilité ?

A. Le container encapsule toutes les dépendances (Python, librairies, versions exactes) dans une image immuable qui s'exécute identiquement sur n'importe quel hôte ✓
B. Docker chiffre le modèle pour le protéger contre le vol de propriété intellectuelle
C. Docker permet de distribuer l'inférence sur plusieurs machines automatiquement

> **Réponse : A** — "Works on my machine" disparaît : l'image Docker garantit que dev, staging et prod utilisent exactement le même environnement.

---

**Q8.** À quel problème répond un registre de modèles (model registry) ?

A. Il stocke les données d'entraînement pour permettre le ré-entraînement futur des modèles
B. Il centralise le versioning des modèles, leurs métadonnées (métriques, hyperparamètres, dataset) et leur cycle de vie (staging, production, archived), facilitant la traçabilité et les rollbacks ✓
C. Il remplace le besoin de tests unitaires en vérifiant automatiquement la qualité des modèles

> **Réponse : B** — Sans registre, savoir quel modèle tourne en prod, avec quelles données il a été entraîné, et comment revenir à une version précédente devient vite ingérable.

---

**Q9.** Quelle est la différence entre un déploiement canary et un déploiement A/B testing ?

A. Canary = rollout progressif vers 100% des utilisateurs pour détecter les problèmes tôt avec possibilité de rollback ; A/B = partage délibéré et durable du trafic pour comparer deux versions et mesurer l'impact ✓
B. Canary = déploiement sur tous les serveurs simultanément ; A/B = déploiement progressif sur un serveur à la fois
C. Canary est pour les modèles ML, A/B testing est pour les applications web classiques

> **Réponse : A** — Canary vise à réduire le risque du déploiement ; A/B testing vise à mesurer l'impact métier — les objectifs et durées sont différents.

---

**Q10.** Pourquoi le CI/CD pour le ML diffère-t-il du CI/CD classique pour une application web ?

A. Le ML n'a pas besoin de tests automatisés car les modèles s'évaluent par leurs métriques
B. En plus du code, il faut versionner et tester les données et les modèles ; les pipelines incluent des étapes d'entraînement, d'évaluation de métriques, et de validation du comportement sur des slices critiques ✓
C. Le CI/CD ML est plus simple car les modèles sont des boîtes noires sans tests unitaires possibles

> **Réponse : B** — Un changement de données peut dégrader les performances sans toucher au code ; les tests de régression ML comparent les métriques entre versions du modèle.

---
