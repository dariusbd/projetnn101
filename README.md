# Neural Net 101

## Structuration du module d'initiation aux NN

**CONCEPTS

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

2. **Spécificité des réseaux de neurones

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

3. **Présentation des architectures

Warren McCulloch et Walter Pitts ont proposé un modèle très simple du neurone biologique, connu par la suite sous le nom de neurone artificiel : il possède une ou plusieurs entrées binaires (marche/arrêt) et une sortie binaire. 

Ces deux scientifiques ont montré que même avec un modèle aussi simplifié, il est possible de construire un réseau de neurones artificiels qui calcule n'importe quelle proposition logique.

Vous pouvez facilement imaginer comment ces réseaux peuvent être combinés pour calculer des expressions complexes. 
Le Perceptron est l'une des architectures ANN les plus simples, inventée en 1957 par Frank Rosenblatt. Il est basé sur un neurone artificiel légèrement différent appelé unité logique à seuil (TLU), ou parfois unité à seuil linéaire (LTU).
![Image](https://raw.githubusercontent.com/dariusbd/projetnn101/main/perceptron.PNG)



Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/dariusbd/projetnn101/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
