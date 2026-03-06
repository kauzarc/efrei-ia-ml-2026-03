# Idées de questions QCM

Questions orientées compréhension conceptuelle (pas mémorisation de formules).

---

## J1AM — Fondations mathématiques & outils Python

1. Pourquoi préfère-t-on utiliser NumPy plutôt que des boucles Python pour les opérations sur les tableaux ?
2. Qu'est-ce que le broadcasting en NumPy ?
3. À quoi sert le produit scalaire entre deux vecteurs géométriquement ?
4. Quelle est la différence entre une matrice et un vecteur du point de vue d'une transformation linéaire ?
5. Qu'est-ce qu'un gradient ?
6. Pourquoi la règle de la chaîne (chain rule) est-elle centrale en deep learning ?
7. Quelle est la différence entre un DataFrame Pandas et un tableau NumPy ?
8. Pourquoi le maximum de vraisemblance conduit-il à minimiser l'erreur quadratique moyenne (MSE) dans le cas d'un bruit gaussien ?

---

## J1PM — Principes du ML & Régression linéaire

1. Quelle est la différence entre apprentissage supervisé, non supervisé et par renforcement ?
2. Pourquoi ne peut-on pas évaluer un modèle uniquement sur ses données d'entraînement ?
3. Que représente la fonction de perte (loss function) dans un problème d'apprentissage ?
4. Quel est l'effet d'un taux d'apprentissage (learning rate) trop élevé lors de l'optimisation ?
5. Quelle est la différence entre le gradient descent en batch, stochastique et mini-batch ?
6. Pourquoi utilise-t-on le mini-batch en pratique plutôt que le batch ou le SGD pur ?
7. Pourquoi ne peut-on pas toujours utiliser la solution analytique pour la régression linéaire ?
8. Qu'est-ce que la convexité d'une fonction de perte nous garantit-elle pour l'optimisation ?

---

## J2AM — Classification & Perceptron

1. Quelle est la différence fondamentale entre un problème de régression et un problème de classification ?
2. Pourquoi la fonction `sign` ne peut-elle pas être utilisée directement dans une descente de gradient ?
3. Quel est le rôle d'une fonction d'activation non-linéaire dans un réseau de neurones ?
4. Pourquoi ReLU est-elle préférée à d'autres fonctions non-linéaires en pratique ?
5. Quelle est la limitation fondamentale du perceptron de Rosenblatt à une seule couche ?
6. Pourquoi le problème XOR est-il un cas emblématique en deep learning ?
7. À quoi sert la fonction softmax dans une classification multi-classe ?
8. Que signifie que la sortie du softmax est une distribution de probabilité ?
9. Pourquoi ajouter des couches cachées avec des non-linéarités permet-il de résoudre XOR ?

---

## J2PM — Validation & Généralisation

1. Comment distingue-t-on l'overfitting de l'underfitting en pratique ?
2. Pourquoi un modèle avec 100% de précision en entraînement n'est-il pas forcément bon ?
3. À quoi sert le jeu de validation, distinct du jeu de test ?
4. Qu'est-ce que le compromis biais-variance ?
5. Un modèle trop simple souffre-t-il plutôt de biais ou de variance ?
6. Pourquoi utilise-t-on la validation croisée (k-fold) plutôt qu'un simple split train/test ?
7. Quelle est la différence entre un hyperparamètre et un paramètre appris ?
8. Pourquoi ne doit-on jamais utiliser le jeu de test pour sélectionner un modèle ?

---

## J3AM — Autograd & Rétropropagation

1. Pourquoi a-t-on besoin de la rétropropagation pour entraîner un réseau profond ?
2. Qu'est-ce que le "problème de l'attribution du crédit" (credit assignment problem) ?
3. Qu'est-ce qu'un graphe de calcul (computation graph) ?
4. Quelle est la différence entre la passe avant (forward pass) et la passe arrière (backward pass) ?
5. Pourquoi empiler des couches linéaires sans non-linéarités entre elles n'a aucun intérêt ?
6. Comment les gradients se propagent-ils lorsqu'un nœud a plusieurs chemins entrants ?
7. Pourquoi faut-il remettre les gradients à zéro (`zero_grad`) avant chaque itération d'entraînement ?

---

## J3PM — PyTorch & CNN

1. Quelle est la différence entre un tenseur PyTorch et un tableau NumPy ?
2. À quoi sert le flag `requires_grad` dans PyTorch ?
3. Pourquoi un MLP (réseau dense) est-il peu adapté au traitement d'images ?
4. Qu'est-ce que le partage de poids dans une convolution ?
5. Quelle est la différence entre stride et padding dans une couche de convolution ?
6. À quoi sert le max pooling dans un CNN ?
7. Que représente un "feature map" en sortie d'une couche convolutive ?
8. Comment les CNN apprennent-ils des représentations hiérarchiques des images ?
9. Quelles sont les étapes typiques d'une boucle d'entraînement PyTorch ?

---

## J4AM — Tokenisation & Embeddings

1. Quel est le principal inconvénient d'une tokenisation au niveau du mot (word-level) ?
2. Pourquoi la tokenisation subword (BPE) est-elle le compromis préféré aujourd'hui ?
3. Quel est le problème principal de l'encodage one-hot pour représenter des mots ?
4. Qu'est-ce qu'un embedding dense ?
5. Qu'est-ce que l'hypothèse distributionnelle sur laquelle repose Word2Vec ?
6. Que signifie que deux mots ont une similarité cosinus proche de 1 ?
7. Pourquoi des analogies comme "roi - homme + femme ≈ reine" émergent-elles des embeddings ?
8. Quel est le problème de la polysémie avec les embeddings statiques comme Word2Vec ?
9. Quelle information importante un sac de mots (bag-of-words) ignore-t-il ?

---

## J4PM — Transformers & Fine-tuning

1. Quel avantage l'attention offre-t-elle par rapport aux RNNs pour traiter des séquences ?
2. Dans le mécanisme d'attention, à quoi correspondent les Query, Key et Value ?
3. Quelle est la différence entre un Transformer encodeur (BERT) et décodeur (GPT) ?
4. Pourquoi l'encodage positionnel est-il nécessaire dans un Transformer ?
5. Qu'est-ce que le transfer learning ?
6. Quelle est la différence entre pre-training et fine-tuning ?
7. Pourquoi les couches basses d'un modèle pré-entraîné sont-elles souvent gelées lors du fine-tuning ?
8. Quel problème la RAG (Retrieval-Augmented Generation) résout-elle ?
9. Dans quel cas faut-il préférer le fine-tuning au prompt engineering ?

---

## J5AM — MLOps & Industrialisation

1. Quel est le principal risque d'utiliser Pickle pour sérialiser un modèle en production ?
2. Quel est l'avantage du format ONNX par rapport à un format natif PyTorch ?
3. Pourquoi faut-il éviter de charger le modèle dans la fonction de route d'une API ?
4. Qu'est-ce que le "train/serve skew" ?
5. Qu'est-ce que la quantification d'un modèle ?
6. Quelle est la différence entre data drift et concept drift ?
7. Pourquoi containeriser un modèle avec Docker améliore-t-il la reproductibilité ?
8. À quel problème répond un registre de modèles (model registry) ?
9. Quelle est la différence entre un déploiement canary et un déploiement A/B testing ?
10. Pourquoi le CI/CD pour le ML diffère-t-il du CI/CD classique pour une application web ?
