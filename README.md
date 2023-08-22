# Classification of pet expressions (computer vision)

This code trains a deep-learning model to perform classification on a particularly difficult computer vision dataset: https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset?datasetId=3546787&sortBy=voteCount

This dataset poses particular issues, as it requires a more nuanced understanding of the images to detect an animals expression vs a classification task such as cats vs dogs.

There are inherent biases in the dataset which also pose issues. For example, a much greater proportion of the training images labeled as “angry” are cats vs the images labelled as “happy”. The result of this is that images of a cat are more likely to labelled as “angry” irrespective of the facial expression, thus introducing bias into the model.

The dataset contains 4 labels, (which have been balanced via data augmentation) meaning that any accuracy scores >25% should be considered to be statistically significant.

A baseline neural network (14,988,996 parameters) was trained and achieved a test accuracy score of 23.7%. This is approximately the same as the accuracy score we would expect from randomly assigning classifications to the images, therefore demonstrating that the model is unable to successfully classify the images in the dataset.

Subsequently, a larger transfer-learning based model (50,174,724 parameters) which utilises the ResNet50V2 pre-trained convolutional base was trained and achieved a test accuracy score of 44.74%. This represents a marked improvement and a statistically significant result, whilst still highlighting the difficulty of the classification task.

In a production environment: the model hyper parameters should be fine-tuned to maximise performance, regularisation techniques should be employed to limit overfitting and the training dataset should be revisited to ensure that any inherent biases are addressed.
