## Structuration du module d'initiation aux NN

**CONCEPTS**

1. **Motivation pour l’apprentissage en général**

De nombreux articles ont démontré que le Machine Learning était capable de réalisations époustouflantes qu’aucune autre technique ne pouvait atteindre. Depuis quelques années maintenant, le Machine Learning (ML) a conquis le secteur : il est désormais au cœur de la magie des produits high-tech d'aujourd'hui, par exemple en classant les résultats de vos recherches sur Internet, en battant le champion du monde au jeu de Go, etc. Avant que vous ne le sachiez, il conduira votre voiture.

Peut-être souhaitez-vous donner à votre robot maison un cerveau qui lui est propre Ou peut-être votre entreprise possède-t-elle des tonnes de données (journaux d'utilisateurs, données financières, etc.). Le ML peut vous aider par exemple à : Segmenter les clients et trouver la meilleure stratégie marketing pour chaque groupe, recommander des produits pour chaque client en fonction de ce que des clients similaires ont acheté, détecter les transactions susceptibles d'être frauduleuses, prévoir les revenus de l'année suivante.

Le ML est l’art de programmer des ordinateurs pour qu'ils puissent apprendre à partir de données. Une définition plus générale dit que le ML est un domaine d'étude qui donne aux ordinateurs la capacité d'apprendre sans être explicitement programmés.

Le Machine Learning n'est pas n'est pas qu'un fantasme futuriste. En fait, il existe depuis plusieurs dizaines d'années dans certaines applications spécialisées, comme le filtre anti-spam. Si nous voulons écrire un filtre Anti-spam en utilisant la technique de programmation traditionnelle :

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/images/Block-Diagram-of-Spam-Filter.ppm.png)

Il existe différents de systèmes de ML
-	Des systèmes qui sont formé ou non par la supervision humaine (supervised, unsupervised, semisupervised, and Reinforcement Learning)
-	Des systèmes qui peuvent apprendre progressivement (online vs batch learning)
-	Des systèmes qui fonctionnent en comparant des nouveaux points de données à des points de données connus (instance-based versus model-based learning).

2. **Spécificité des réseaux de neurones**

Les oiseaux nous inspirés à créer l’avion. Et depuis, d'innombrables autres inventions ont été inspirées par la nature. Il semble logique, alors, de regarder l'architecture du cerveau pour s'inspirer de la façon de construire une machine intelligente. C'est l'idée clé qui a donné naissance aux réseaux de neurones artificiels (ANN).

Les ANN sont au cœur même du Deep Learning. Ils sont polyvalents, puissants et évolutifs, ce qui les rend idéaux pour s'attaquer à des tâches de Machines Learning vastes et très complexes, telles que la classification de milliards d'images (p. ex. Google Images), l'alimentation de services de reconnaissance vocale (p. ex. Siri d'Apple), la recommandation des meilleures vidéos à regarder à des centaines de millions d'utilisateurs chaque jour (p. ex. YouTube).

Les ANN existent depuis longtemps : ils ont été présentés pour la première fois en 1943 par le neurophysiologiste Warren McCulloch et le mathématicien Walter Pitts.
Ces deux scientifiques ont présenté un modèle de calcul simplifié de la façon dont les neurones biologiques pouvaient travailler ensemble dans le cerveau des animaux pour effectuer des calculs complexes en utilisant la logique propositionnelle. Il s'agissait de la première architecture de réseau neuronal artificiel.

Nous assistons aujourd'hui à une nouvelle vague d'intérêt pour les ANNs. Il y a quelques bonnes raisons de croire que cette vague est différente et qu'elle aura un impact beaucoup plus profond sur nos vies :
-	Il existe aujourd'hui une énorme quantité de données disponibles pour former les réseaux neuronaux, et les ANN surpassent fréquemment les autres techniques ML sur des problèmes très vastes et complexes.
-	L'augmentation considérable de la puissance de calcul depuis les années 1990 permet désormais de former de grands réseaux neuronaux en un temps raisonnable. Certaines limites théoriques des ANN se sont avérées bénignes dans la pratique. Par exemple, de nombreuses personnes pensaient que les algorithmes d'apprentissage des ANN étaient condamnés parce qu'ils étaient susceptibles de rester bloqués dans des optima locaux, mais il s'avère que cela est plutôt rare dans la pratique (ou lorsque c'est le cas, ils sont généralement assez proches de l'optimum global).

Des produits étonnants basés sur les ANN font régulièrement la une des journaux, ce qui attire de plus en plus l'attention et le financement sur eux, ce qui entraîne de plus en plus de progrès, et des produits encore plus étonnants.
Warren McCulloch et Walter Pitts ont proposé un modèle très simple du neurone biologique, connu par la suite sous le nom de neurone artificiel : il possède une ou plusieurs entrées binaires (marche/arrêt) et une sortie binaire.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/images/reseaux_neurones_feed_forwarded_2.png)

Ils ont montré que même avec un modèle aussi simplifié, il est possible de construire un réseau de neurones artificiels qui calcule n'importe quelle proposition logique.

3. **Présentation des architectures**

Warren McCulloch et Walter Pitts ont proposé un modèle très simple du neurone biologique, connu par la suite sous le nom de neurone artificiel : il possède une ou plusieurs entrées binaires (marche/arrêt) et une sortie binaire. 

Ces deux scientifiques ont montré que même avec un modèle aussi simplifié, il est possible de construire un réseau de neurones artificiels qui calcule n'importe quelle proposition logique.

Vous pouvez facilement imaginer comment ces réseaux peuvent être combinés pour calculer des expressions complexes. 
Le Perceptron est l'une des architectures ANN les plus simples, inventée en 1957 par Frank Rosenblatt. Il est basé sur un neurone artificiel légèrement différent appelé unité logique à seuil (TLU), ou parfois unité à seuil linéaire (LTU).

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/images/perceptron.PNG)

Cependant, il s'avère que certaines des limitations des perceptrons peuvent être éliminées en empilant plusieurs perceptrons. L'ANN qui en résulte est appelé Perceptron multicouche (MLP). En particulier, un MLP peut résoudre le problème XOR.

Un MLP est composé d'une couche d'entrée, une ou plusieurs couches d'ULT, appelées couches cachées, et d'une couche finale d'ULT appelée couche de sortie. Les couches proches de la couche d'entrée sont généralement appelées les couches inférieures, et celles proches des sorties sont généralement appelées les couches supérieures.

Lorsqu'un ANN contient une pile profonde de couches cachées, on l'appelle un réseau neuronal profond (DNN). Premièrement, les MLP peuvent être utilisés pour des tâches de régression. Si vous voulez prédire une seule valeur, vous n'avez besoin que d'un seul neurone de sortie : sa sortie est la valeur prédite. Pour une régression multivariée (c'est-à-dire pour prédire plusieurs valeurs à la fois), vous avez besoin d'un neurone de sortie par dimension de sortie.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/images/MLP.PNG)

Les MLP peuvent également être utilisés pour des tâches de classification. Pour un problème de classification binaire, il suffit d'un seul neurone de sortie utilisant la fonction d'activation logistique : la sortie sera un nombre compris entre 0 et 1, que vous pouvez interpréter comme la probabilité estimée de la classe positive. Ils peuvent également gérer facilement les tâches de classification binaire multi-label Par exemple, vous pourriez avoir un système de classification d'e-mails qui prédit s'il s'agit d'un e-mail urgent ou non urgent.

Un autre type de réseaux de neurones dit convolutifs (CNN) sont issus de l'étude du cortex visuel du cerveau et sont utilisés dans la reconnaissance d'images depuis les années 1980. L'élément constitutif le plus important d'un CNN est la couche convolutive.
Les neurones de la première couche convolutive ne sont pas connectés à chaque pixel de l'image d'entrée, mais uniquement aux pixels situés dans leur champ récepteur.
À son tour, chaque neurone de la deuxième couche convolutive est connecté uniquement aux neurones situés dans un petit rectangle de la première couche.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/images/CNN.PNG)

Cette architecture permet au réseau de se concentrer sur de petites caractéristiques de bas niveau dans la première couche cachée, puis de les assembler en caractéristiques de plus haut niveau dans la couche cachée suivante, et ainsi de suite. Cette structure hiérarchique est courante dans les images du monde réel, et c'est l'une des raisons pour lesquelles les réseaux CNN fonctionnent si bien pour la reconnaissance d'images.

4. **Principe d’optimisation**

Une bonne compréhension du fonctionnement des choses peut aider à trouver rapidement le modèle approprié, le bon algorithme d'apprentissage à utiliser. La régression linéaire, l’un des modèles les plus simples qui soient peut être abordé de deux manières très différentes :
-	Soit par l’utilisation d'une équation directe "à forme fermée" qui calcule directement les paramètres du modèle qui s'adaptent le mieux à l'ensemble d'apprentissage
-	Soit par l'utilisation d'une approche d'optimisation itérative, appelée descente de gradient (GD).

Nous porterons notre attention sur la seconde approche, celle de l’optimisation itérative. 
La descente par gradient est un algorithme d'optimisation très générique capable de trouver des solutions optimales à un large éventail de problèmes. 

L'idée générale de la descente par gradient est de modifier les paramètres de manière itérative afin de minimiser une fonction de coût. Supposons que vous soyez perdu en montagne dans un brouillard dense ; vous ne pouvez sentir que la pente du sol sous vos pieds. Une bonne stratégie pour atteindre rapidement le fond de la vallée consiste à descendre dans la direction de la pente la plus raide. C'est exactement ce que fait la descente par gradient.

Un paramètre important de la descente par gradient est la taille des étapes, déterminée par l'hyperparamètre du taux d'apprentissage. Si le taux d'apprentissage est trop faible, l'algorithme devra passer par de nombreuses itérations.
Les différents algorithmes de la Descente de gradient (GD):

-	Descente de gradient par lots

Pour mettre en œuvre la descente par gradient, vous devez calculer le gradient de la fonction de coût en fonction de chaque paramètre du modèle θj. En d'autres termes, vous devez calculer de combien la fonction de coût changera si vous modifiez θj juste un peu. C'est ce qu'on appelle une dérivée partielle. Le principal problème de la descente de gradient par lots est qu'elle utilise l'ensemble de l'apprentissage pour calculer les gradients à chaque étape, ce qui la rend très lente lorsque l'ensemble d'apprentissage est important.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/images/batch%20GD.jpg)

-	Descente de gradient stochastique (aléatoire)
L'algorithme de descente en gradient stochastique choisit simplement une instance aléatoire dans l'ensemble d'apprentissage à chaque étape et calcule les gradients en se basant uniquement sur cette instance unique. D'autre part, en raison de sa nature stochastique, cet algorithme est beaucoup moins régulier que l'algorithme de descente de gradient par lots : au lieu de décroître doucement jusqu'à atteindre le minimum, la fonction de coût va rebondir, ne décroissant qu'en moyenne.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/images/GD%20stochastic.PNG)

-	Descente de gradient par mini-lots
Cet algorithme est assez simple à comprendre. A chaque étape, la GD par mini-lots calcule les gradients sur de petits ensembles aléatoires d'instances appelés mini-lots. Le principal avantage de la GD mini-batch par rapport à la GD stochastique est que vous pouvez obtenir un gain de performance grâce à l'optimisation matérielle des opérations matricielles, en particulier lorsque vous utilisez des GPU.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/images/GD%20par%20mini%20lots.PNG)

2. **Représentation des données**

En général, tous les systèmes actuels de Machine Learning utilisent des tenseurs comme structure de données de base. Les tenseurs sont fondamentaux dans ce. Même les données textuelles ou les données d'image sont converties en caractéristiques numériques pour être traitées.
Les tenseurs sont une structure de données spécialisée qui ressemble beaucoup aux tableaux et aux matrices. Dans PyTorch, nous utilisons les tenseurs pour coder les entrées et les sorties d'un modèle, ainsi que les paramètres du modèle.
Nous savons que les tenseurs ont différents types de dimensions tels que la dimension zéro, une dimension et multidimensionnelle.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/images/N-dimensional%20array.png)

- **Vecteurs**

Les vecteurs sont des tenseurs unidimensionnels, et pour les manipuler, plusieurs opérations sont disponibles. Les opérations sur les vecteurs sont de différents types tels que l'opération mathématique, le produit scalaire et le linspace. Les vecteurs jouent un rôle essentiel dans le Deep Learning.
Dans le réseau neuronal du Deep Learning, nous générons des points aléatoires à l'aide de vecteurs ou de tenseurs unidimensionnels. Les opérations suivantes sont effectuées sur le vecteur.
Nous pouvons ajouter, soustraire, multiplier et diviser le tenseur d'un autre tenseur. Voici quelques opérations qui sont effectuées sur des vecteurs avec le résultat attendu.

import torch
A=torch.tensor([1,2,3])
B=torch.tensor([4,5,6])
A+B
B-2
A*B
A*2
B/A

Output:
tensor([5, 7, 9])
tensor([2, 3, 4])
tensor([ 4, 10, 18])
tensor([2, 4, 6])
tensor([4, 2, 2])

- **Matrices**

Un tableau de vecteurs est une matrice, ou tenseur 2D. Une matrice a deux axes (souvent appelés lignes et colonnes). Vous pouvez interpréter visuellement une matrice comme une grille rectangulaire de nombres. Ceci est une matrice Numpy:

Input : 
x = np.array([[5, 78, 2, 34, 0],
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]])
x.ndim 
Output : 
2

- **Tenseurs 3D et tenseurs de plus grande dimension **

Si vous empaquetez de telles matrices dans un nouveau tableau, vous obtenez un tenseur 3D, que vous pouvez interpréter visuellement comme un cube de nombres. Voici un tenseur Numpy 3D:

Input : 
from numpy import array 
T = array([
  [ [1,2,3], [4,5,6], [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29] ],
  ]) 
  print(T.shape) 
  print("3D Tensor T is: ", T) 
  print("Dimension of 3D Tensor T is: ", T.ndim)
  
  Output : 
(3, 3, 3) 
[[[ 1 2 3]
[ 4 5 6]
[ 7 8 9]]
  
[[11 12 13]
[14 15 16]
[17 18 19]]
  
  [[21 22 23] 
  [24 25 26] 
  [27 28 29]]] 
  
  Dimension of 3D Tensor T is : 3
  
Voyez le résultat de l'opération ci-dessus, pour comprendre la structure d'un tenseur 3D. Il s'agit d'une collection de matrices. Ainsi, contrairement à une matrice unique avec deux axes, un tenseur 3-D a trois axes.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/images/T1.PNG)

- **Tenseurs 4-D**

De la même manière que nous obtenons un tenseur 3-D, si certains de ces tenseurs 3-D doivent être groupés, une autre dimension est créée, faisant du tenseur un tenseur 4-D. Ici, vous pouvez voir trois cubes sont matraqués. De tels tenseurs 4-D sont très utiles pour stocker des images pour la reconnaissance d'images dans le deep learning. 

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/images/T2.png)

- **Tenseurs**

De la même manière, nous pouvons avoir des tenseurs de dimension plus élevée. Bien que les tenseurs jusqu'à 4 dimensions soient plus courants. C'est le type de tenseurs lorsque nous devons stocker des données avec encore une autre dimension. Les données vidéo peuvent être un exemple idéal où des tenseurs 5-D sont utilisés.
Les tenseurs sont similaires aux ndarrays de NumPy, sauf qu'ils peuvent fonctionner sur des GPU ou d'autres accélérateurs matériels. En fait, les tenseurs et les tableaux NumPy peuvent souvent partager la même mémoire sous-jacente, ce qui élimine le besoin de copier les données (voir Bridge with NumPy). Les tenseurs sont également optimisés pour la différenciation automatique (nous y reviendrons plus tard dans la section Autograd).
