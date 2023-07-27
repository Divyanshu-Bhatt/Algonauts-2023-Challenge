# Algonauts-2023-Challenge

### Setup
1. Downloading the NSD-Dataset
  ``` 
  cd data
  bash download.sh
  ```

2. Extracting Features
  ```
  bash extract_features 
  ```
  The image is passed through pre-trained ResNet, and the output of Layer 1, 2, 3 and 4 are saved as features for other models. Similarly, the image is given as input to pre-trained ViT model to create CLIP-embeddings.

### Solution
First tried to fine-tune the ResNet by just changing its last layer and training the model. Run
```
bash resnet_train.sh
```

A LeNet is trained with the extracted ResNet features as input.
```
bash lenet_train.sh
```

A CrossVAE was trained which takes extracted ResNet features and CLIP embeddings as input.
```
bash train_CrossVAE.sh
```

### Results
All the experimenets and hyperparameter tuning, and decisions in changing the architecutre were made by seeing the results on only ```subj01``` because the models take a lot of time in training for all the subjects.

1. The Layer 3 extracted features gave the highest correlation when trained a LeNet.
2. Training different models for individual ROIs and concatenating the output of these models gives a better result then training a single model for predicting the whole output vector.
3. CrossVAE was trained only with ResNet Layer 3 features, others were not explored.

| Models | Correlation|
|--------|--------------|
| ResNet | 33.3 |
| LeNet | 47.7 |
| CrossVAE | 47.6 |




