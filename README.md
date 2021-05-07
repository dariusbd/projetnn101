# Neural Net 101

## Structuration du module d'initiation aux NN

**CONCEPTS**

1. **Motivation pour l’apprentissage en général**

De nombreux articles ont démontré que le Machine Learning était capable de réalisations époustouflantes qu’aucune autre technique ne pouvait atteindre.
Depuis quelques années maintenant, le Machine Learning (ML) a conquis le secteur : il est désormais au cœur de la magie des produits high-tech d'aujourd'hui, par exemple en classant les résultats de vos recherches sur Internet, en battant le champion du monde au jeu de Go, etc. Avant que vous ne le sachiez, il conduira votre voiture.

Peut-être souhaitez-vous donner à votre robot maison un cerveau qui lui est propre Ou peut-être votre entreprise possède-t-elle des tonnes de données (journaux d'utilisateurs, données financières, etc.).  Le ML peut vous aider par exemple à : Segmenter les clients et trouver la meilleure stratégie marketing pour chaque groupe, recommander des produits pour chaque client en fonction de ce que des clients similaires ont acheté, détecter les transactions susceptibles d'être frauduleuses, prévoir les revenus de l'année suivante.

Le ML est l’art de programmer des ordinateurs pour qu'ils puissent apprendre à partir de données. Une définition plus générale dit que le ML est un domaine d'étude qui donne aux ordinateurs la capacité d'apprendre sans être explicitement programmés.

Le Machine Learning n'est pas n'est pas qu'un fantasme futuriste. En fait, il existe depuis plusieurs dizaines d'années dans certaines applications spécialisées, comme le filtre anti-spam. Si nous voulons écrire un filtre Anti-spam en utilisant la technique de programmation traditionnelle : 

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/Block-Diagram-of-Spam-Filter.ppm.png)

Tout d'abord, il faut voir à quoi ressemble un spam, ensuite il faut remarquer les mots ou expressions (comme "4U", "carte de crédit", "gratuit" et "incroyable") qui ont tendance à revenir souvent dans le sujet. Puis vous écrivez un algorithme de détection pour chacun des modèles que vous avez remarqués, et votre programme signalera les emails comme spam si un certain nombre de ces modèles sont détectés. A la fin, vous testez votre programme et répétez ces étapes jusqu'à ce qu'il soit suffisamment efficace

Comme le problème n'est pas trivial, votre programme deviendra probablement une longue liste de règles complexes, assez difficile à maintenir.
En revanche, un filtre anti-spam basé sur des techniques de ML apprend automatiquement quels mots et phrases sont de bons indicateurs de spam en détectant des modèles de mots inhabituellement fréquents dans les exemples de spam. Le programme est beaucoup plus court, plus facile à maintenir et très probablement plus précis

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/anti%20spam.png)

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

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/reseaux_neurones_feed_forwarded_2.png)

Ils ont montré que même avec un modèle aussi simplifié, il est possible de construire un réseau de neurones artificiels qui calcule n'importe quelle proposition logique.

3. **Présentation des architectures**

Warren McCulloch et Walter Pitts ont proposé un modèle très simple du neurone biologique, connu par la suite sous le nom de neurone artificiel : il possède une ou plusieurs entrées binaires (marche/arrêt) et une sortie binaire. 

Ces deux scientifiques ont montré que même avec un modèle aussi simplifié, il est possible de construire un réseau de neurones artificiels qui calcule n'importe quelle proposition logique.

Vous pouvez facilement imaginer comment ces réseaux peuvent être combinés pour calculer des expressions complexes. 
Le Perceptron est l'une des architectures ANN les plus simples, inventée en 1957 par Frank Rosenblatt. Il est basé sur un neurone artificiel légèrement différent appelé unité logique à seuil (TLU), ou parfois unité à seuil linéaire (LTU).

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/perceptron.PNG)

Cependant, il s'avère que certaines des limitations des perceptrons peuvent être éliminées en empilant plusieurs perceptrons. L'ANN qui en résulte est appelé Perceptron multicouche (MLP). En particulier, un MLP peut résoudre le problème XOR.

Un MLP est composé d'une couche d'entrée, une ou plusieurs couches d'ULT, appelées couches cachées, et d'une couche finale d'ULT appelée couche de sortie. Les couches proches de la couche d'entrée sont généralement appelées les couches inférieures, et celles proches des sorties sont généralement appelées les couches supérieures.

Lorsqu'un ANN contient une pile profonde de couches cachées, on l'appelle un réseau neuronal profond (DNN). Premièrement, les MLP peuvent être utilisés pour des tâches de régression. Si vous voulez prédire une seule valeur, vous n'avez besoin que d'un seul neurone de sortie : sa sortie est la valeur prédite. Pour une régression multivariée (c'est-à-dire pour prédire plusieurs valeurs à la fois), vous avez besoin d'un neurone de sortie par dimension de sortie.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/MLP.PNG)

Les MLP peuvent également être utilisés pour des tâches de classification. Pour un problème de classification binaire, il suffit d'un seul neurone de sortie utilisant la fonction d'activation logistique : la sortie sera un nombre compris entre 0 et 1, que vous pouvez interpréter comme la probabilité estimée de la classe positive. Ils peuvent également gérer facilement les tâches de classification binaire multi-label Par exemple, vous pourriez avoir un système de classification d'e-mails qui prédit s'il s'agit d'un e-mail urgent ou non urgent.

Un autre type de réseaux de neurones dit convolutifs (CNN) sont issus de l'étude du cortex visuel du cerveau et sont utilisés dans la reconnaissance d'images depuis les années 1980. L'élément constitutif le plus important d'un CNN est la couche convolutive.
Les neurones de la première couche convolutive ne sont pas connectés à chaque pixel de l'image d'entrée, mais uniquement aux pixels situés dans leur champ récepteur.
À son tour, chaque neurone de la deuxième couche convolutive est connecté uniquement aux neurones situés dans un petit rectangle de la première couche.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/CNN.PNG)

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

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/batch%20GD.jpg)

-	Descente de gradient stochastique (aléatoire)
L'algorithme de descente en gradient stochastique choisit simplement une instance aléatoire dans l'ensemble d'apprentissage à chaque étape et calcule les gradients en se basant uniquement sur cette instance unique. D'autre part, en raison de sa nature stochastique, cet algorithme est beaucoup moins régulier que l'algorithme de descente de gradient par lots : au lieu de décroître doucement jusqu'à atteindre le minimum, la fonction de coût va rebondir, ne décroissant qu'en moyenne.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/GD%20stochastic.PNG)

-	Descente de gradient par mini-lots
Cet algorithme est assez simple à comprendre. A chaque étape, la GD par mini-lots calcule les gradients sur de petits ensembles aléatoires d'instances appelés mini-lots. Le principal avantage de la GD mini-batch par rapport à la GD stochastique est que vous pouvez obtenir un gain de performance grâce à l'optimisation matérielle des opérations matricielles, en particulier lorsque vous utilisez des GPU.

![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/GD%20par%20mini%20lots.PNG)


**ASPECTS PRATIQUE**

1. **Intro à Numpy**



2. **Représentation des données**



3. **Intro à pandas**



4. **Intro deep learning**



5. **Intro Computer Vision** 



[Link](url) and ![Image](src)
