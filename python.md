# Python

## Introduction à Numpy

### Qu'est-ce que NumPy ?

NumPy est une bibliothèque Python utilisée pour travailler avec des tableaux.
Elle dispose également de fonctions permettant de travailler avec l'algèbre linéaire, les transformations de Fourier et les matrices.
NumPy a été créé en 2005 par Travis Oliphant. Il s'agit d'un projet open source que l’on peut utiliser librement. NumPy est l'acronyme de Numerical Python.

### Pourquoi utiliser NumPy ?

En Python, nous avons des listes qui font office de tableaux, mais elles sont lentes à traiter.
NumPy vise à fournir un objet tableau qui est jusqu'à 50 fois plus rapide que les listes traditionnelles de Python.
L'objet tableau dans NumPy est appelé ndarray, il fournit de nombreuses fonctions de soutien qui rendent le travail avec ndarray très facile.
Les tableaux sont très fréquemment utilisés en Data science, où la vitesse et les ressources sont très importantes.

### Pourquoi NumPy est-il plus rapide que les listes ?

Contrairement aux listes, les tableaux NumPy sont stockés à un endroit continu de la mémoire. Les processus peuvent donc y accéder et les manipuler très efficacement.
Ce comportement est appelé localité de référence en informatique.
C'est la principale raison pour laquelle NumPy est plus rapide que les listes. Il est également optimisé pour fonctionner avec les dernières architectures de CPU.
La bibliothèque NumPy permet d’effectuer des calculs numériques avec Python. Elle introduit une gestion facilitée des tableaux de nombres.

Il faut au départ importer le package "numpy" avec l’instruction suivante :
> import numpy as np

### Variables prédéfinies

**Variable pi**
NumPy définit par défaut la valeur de pi.
> np.pi
> ==> 3.141592653589793

### Tableaux - numpy.array()

**Création**

Les tableaux (en anglais, array) peuvent être créés avec **numpy.array()**. On utilise des crochets pour délimiter les listes d’éléments dans les tableaux.
> a = np.array([[1, 2, 3], [4, 5, 6]])

**Affichage**
> a
> ==> array([[1, 2, 3],[4, 5, 6]])

> type(a)
> ==> numpy.ndarray

On voit que l’on a obtenu un objet de type numpy.ndarray

**Accès aux éléments d’un tableau**

Avertissement : Comme pour les listes, les indices des éléments commencent à zéro.

> a[0,1]
> ==> 2

> a[1,2]
> ==> 6

**La fonction numpy.arange()**

> m = np.arange(3, 15, 2)

> m
> ==> array([ 3,  5,  7,  9, 11, 13])

> type(m)
> ==> numpy.ndarray

Noter la différence entre numpy.arange() et range() :
- numpy.arange() retourne un objet de type numpy.ndarray.
- range() retourne un objet de type range.

> n = range(3, 15, 2)

> n
> ==> range(3, 15, 2)

> type(n)
> ==> range

Ceci est également à distinguer d’une liste.

> u = [3, 7, 10]

> type(u)
> ==> list

Il est possible d’obtenir des listes en combinant list et range().

> list(range(3, 15, 2))
> ==> [3, 5, 7, 9, 11, 13]

numpy.arange() accepte des arguments qui ne sont pas entiers.
> np.arange(0, 11*np.pi, np.pi)
> ==> array([  0.        ,   3.14159265,   6.28318531,   9.42477796,
        12.56637061,  15.70796327,  18.84955592,  21.99114858,
        25.13274123,  28.27433388,  31.41592654])

**La fonction numpy.linspace()**

numpy.linspace() permet d’obtenir un tableau 1D allant d’une valeur de départ à une valeur de fin avec un nombre donné d’éléments.
> np.linspace(3, 9, 10)
array([ 3.       ,  3.66666667,  4.33333333,  5.        ,  5.66666667,
        6.33333333,  7.        ,  7.66666667,  8.33333333,  9.        ])
