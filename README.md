# Membership Inference Attack

## Data Generation

## Shadow Model

## Attack Model

*Note* :

- `transform` is to be preformed on images
- `collate_fn` is used in `Dataloader` to perform extra tranformation of data

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

### Augmentaion

1. For each data, calculate a vector of size (len(tran)*time) of 0 (augmented data classified correctly) and 1 (
   augmented data classified incorrectly)
2. clustering of vectors calculated (Kmeans)

**Warning** :

- This method is unsupervised, training of Shadow Model is unnuecessary
- The transformation methods used should be tailored to dataset

```python
attack_model = Augmentation(device, trans, times, transform, collate_fn, batch_size)
# show : TSNE visualization of vectors calculated
attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42), show=True)
```

### Boundary Distance

1. For each data, use [Carlini Wagner Attack](https://arxiv.org/abs/1608.04644)
   of [cleverhans](https://github.com/cleverhans-lab/cleverhans/blob/master/cleverhans/torch/attacks/carlini_wagner_l2.py)
   to generate adversarial example associated
2. Calculate the distance between the data and the adversarial example associated as the distance to decision boundary
3. Using the fact that the larger the distance, the more probable that it is used when training
4. Find two thresholds that maximizes the accuracy and precision

**Warning** :

- Only valable for continuous data
- Very time-consuming

```python
attack_model = Boundary(shadow_models, device, classes, transform)
# historam 
attack_model.train(show=True)
attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42))
```

### Noise

1. For each data, try adding gaussien noise with stddev in input stddev list
2. Calculate the number of times when it is classified corretly
3. Using the fact that the larger the number of times when it is classified corretly, the more probable that it is used
   when training
4. Find two thresholds that maximizes the accuracy and precision

**Warning** :

- Only valable for continuous data
- should manually set the range of valable
- The stddev should be tailored to dataset

```python
attack_model = Noise(shadow_models, stddev, device, transform)
# historam
attack_model.train(show=True)
attack_model.evaluate(target, *train_test_split(target_X, target_Y, test_size=0.5, random_state=42))
```

# Attack Pipeline

# Demonstration

# Defense Methods