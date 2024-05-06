# gan-cgan

__Encadrant: SCHNEIDER Léo__ <br>
__Étudiant: Victor Ludvig__

#### Partie 1
Dans la première partie, un GAN conditionnel  est implémenté pour générer des images de chiffres à partir de vecteurs gaussiens. Les images d'entrainement sont issues du MNIST dataset.

#### Partie 2
Dans la deuxième partie, un GAN conditionnel est implémenté pour générer des images d'imeubles à partir de plans d'imeubles. <br>
Le discriminateur prend en entrée le couple d'images (plan, imeuble), et prédit si c'est le couple est réel ou généré. <br>
La MSE loss est utilisée pour la loss du GAN, au lieu de la BCE loss, probablement pour avoir une convergence plus simple. Cependant ce n'est pas idéal et conduit à des résultats flous.<br>
Une pixel-wise L1 loss est rajoutée pour le générateur en plus de la cGAN loss classique, afin de prédir les basses fréquences. Cependant, puisque la MSE loss est utilisée pour la cGAN loss du PatchGan, l'utilisation de la loss L1 est probablement redondante, car la MSE loss est connue pour capter les hautes fréquences: 
```
It is well known that the L2 loss – and L1, see Fig-
ure 4 – produces blurry results on image generation prob-
lems [34]. Although these losses fail to encourage high-frequency crispness, in many cases they nonetheless accu-
rately capture the low frequencies.
```
(papier original du PatchGan, p3-4)