# Membership Inference Attack

- Full demonstration of the complete attack pipeline (train/attack/evaluate) in 3 domain (table, NLP, Image) (Classification)
- Multi-Threading
- Easy to use API of 4 attack methods

## Data Generation

Naïve Hill-Climbing Search of all space of possible input of a model, proposed by [R. Shokri](https://arxiv.org/abs/1610.05820), trying to get the data for which the model gives the higher confidence than threshold, if not, changes the data randomly and reiterate.

**Warning** :

Very Costly, should only be used when the dimension of data is relatively small and the data consists only bool, int or float.

## Shadow Model

Train the Shadow Models / reuse the trained Shadow Models to infer (Scikit-Learn or torch) confidence vector for the data used to train the Shadow Models.

This class is integrated into the Confidence Vector Attack, Boundary Attack and Noise Attack. (Augmentation Attack is unsupervised)

## Attack Model

*Note* :

- `transform` is to be preformed on images
- `collate_fn` is used in `Dataloader` to perform extra transformation of data

### Confidence Vector

Direct Classification of Confidence Vectors by a simple MLP defined in `MIA.utils.attackmodel`

**Warning** :

when topx = -1, the whole Confidence Vector will be used, an attack model will be trained for each class

when topx = k >= 1, k most big probabilities of each Confidence Vector will be used, only one attack model will be
trained

```python
attack_model = ConfVector(shadow_models, attack_nepoch, device, topx, transform)
attack_model.train()
# show the 3D distribution of Confidence Vectors (topx forced to 3)
# valable for Confidence Vectors longer than 3
attack_model.show()
# evaluate on shadowmodel
attack_model.evaluate()
# evaluate on target
attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42))

```

### Augmentation

1. For each data, calculate a vector of size (len(tran)*time) of 0 (augmented data classified correctly) and 1 (augmented data classified incorrectly)
2. clustering of vectors calculated (Kmeans)

**Warning** :

- This method is unsupervised, training of Shadow Model is unnecessary
- The transformation methods used should be tailored to dataset

```python
attack_model = Augmentation(device, trans, times, transform, collate_fn, batch_size)
# show : TSNE visualization of vectors calculated
attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42), show=True)
```

### Boundary Distance

1. For each data, use [Carlini Wagner Attack](https://arxiv.org/abs/1608.04644) of [cleverhans](https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/torch/attacks/carlini_wagner_l2.py)
   to generate adversarial example associated
2. Calculate the distance between the data and the adversarial example associated as the distance to decision boundary
3. Using the fact that the larger the distance, the more probable that it is used when training
4. Find two thresholds that maximizes the accuracy and precision

**Warning** :

- Only valid for continuous data
- Very time-consuming

```python
attack_model = Boundary(shadow_models, device, classes, transform)
# historam
attack_model.train(show=True)
attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42))
```

### Noise

1. For each data, try adding gaussian noise with stddev in input stddev list
2. Calculate the number of times when it is classified correctly
3. Using the fact that the larger the number of times when it is classified correctly, the more probable that it is used
   when training
4. Find two thresholds that maximizes the accuracy and precision

**Warning** :

- Only valid for continuous data
- should manually set the range of valid
- The stddev should be tailored to dataset

```python
attack_model = Noise(shadow_models, stddev, device, transform)
# historam
attack_model.train(show=True)
attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42))
```

# Defense Methods

## [Dropout](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

The effectiveness of this method is related to reducing overfitting.

## L2 regularization

Edit the optimizer by adding the parameters of weight_decay  to add L2 regularization.

The effectiveness of this method is related to reducing overfitting.

## [Mix-Up](https://doi.org/10.48550/arXiv.1710.09412)

`mixup_data(x, y, alpha, device)`  takes two data and mix by a coefficient of beta distribution beta(alpha,alpha)

**Warning** :

The loss function must be replaced by `mixup_criterion`

The effectiveness of this method could be find in [this paper](https://link.springer.com/chapter/10.1007/978-3-030-93206-0_3).

## Smart-Noise

`mix(X, Y, ratio)`

More information on the parameters of the _Synthesizer_, please refer to library [Smart-Noise](https://smartnoise.org)

## [MemGuard](https://doi.org/10.48550/arXiv.1909.10594)

`memguard(scores,epsilon)` is the simplified version of MemGuard proposed by [C.C.Christopher](https://github.com/cchoquette/membership-inference)

```python
# better perform softmax before memguard
output_in=F.softmax(output_in, dim=-1)
# memguard use numpy
output_in = memguard(output_in.cpu().numpy())
```
