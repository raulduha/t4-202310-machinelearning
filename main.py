import tensorflow as tf
from emnist import extract_training_samples, extract_test_samples
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Extraemos dataset train y test
train_images, train_labels = extract_training_samples('letters')
test_images, test_labels = extract_test_samples('letters')

#normalizar y reshape
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images_number, train_images_height, train_images_width = train_images.shape
train_images_size = train_images_height * train_images_width

train_images = train_images.reshape(train_images_number, train_images_height, train_images_width, 1)

test_images_number, test_images_height, test_images_width = test_images.shape
test_images_size = test_images_height * test_images_width

test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)

# transformamos labels
number_of_classes = 37

train_labels = tf.keras.utils.to_categorical(train_labels, number_of_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, number_of_classes)

# PCA reduccion dim.
n_components = [128, 64, 32, 16, 8]
pca_results_train = []
pca_results_test = []

for n in n_components:
    flattened_train_images = train_images.reshape(train_images.shape[0], train_images_size)
    flattened_test_images = test_images.reshape(test_images.shape[0], test_images_size)
    
    pca = PCA(n_components=n)
    train_images_pca = pca.fit_transform(flattened_train_images)
    test_images_pca = pca.transform(flattened_test_images)
    
    pca_results_train.append(train_images_pca)
    pca_results_test.append(test_images_pca)

#  umap reduccion dim.
n_neighbors = 15
umap_results_train = []
umap_results_test = []

for n in n_components:
    flattened_train_images = train_images.reshape(train_images.shape[0], train_images_size)
    flattened_test_images = test_images.reshape(test_images.shape[0], test_images_size)
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n)
    train_images_umap = reducer.fit_transform(flattened_train_images)
    test_images_umap = reducer.transform(flattened_test_images)
    
    umap_results_train.append(train_images_umap)
    umap_results_test.append(test_images_umap)

# entrenamos y evaluamos con NN
def nearest_neighbor_classifier(train_features, train_labels, test_features):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_features, train_labels)
    predictions = knn.predict(test_features)
    return predictions

def display_sample_images(images, labels, predictions):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axes = axes.ravel()

    for i in range(10):
        axes[i].imshow(images[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"True: {np.argmax(labels[i])}\nPred: {predictions[i]}")
        axes[i].axis('off')

    plt.show()

original_predictions = nearest_neighbor_classifier(flattened_train_images, np.argmax(train_labels, axis=1), flattened_test_images)
original_accuracy = accuracy_score(np.argmax(test_labels, axis=1), original_predictions)

pca_accuracies = []
umap_accuracies = []

for i, n in enumerate(n_components):
    pca_accuracy = accuracy_score(np.argmax(test_labels, axis=1), nearest_neighbor_classifier(pca_results_train[i], np.argmax(train_labels, axis=1), pca_results_test[i]))
    umap_accuracy = accuracy_score(np.argmax(test_labels, axis=1), nearest_neighbor_classifier(umap_results_train[i], np.argmax(train_labels, axis=1), umap_results_test[i]))
    
    pca_accuracies.append(pca_accuracy)
    umap_accuracies.append(umap_accuracy)
    
    if n == 8:
        display_sample_images(test_images, test_labels, nearest_neighbor_classifier(umap_results_train[i], np.argmax(train_labels, axis=1), umap_results_test[i]))

# resultados y conclusion
categories = [str(n) for n in n_components]

# plt
def plot_results(categories, accuracies, title):
    plt.figure(figsize=(8, 6))
    plt.bar(categories, accuracies, color='b', alpha=0.7)
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim([0, 1])
    plt.show()

# accuracy plot de pca
plot_results(categories, pca_accuracies, 'Accuracy with PCA')

# accuracy plot de umap
plot_results(categories, umap_accuracies, 'Accuracy with UMAP')

# Print results para resultados del infomre
print("Resultados:")
print("--------")
print("Original accuracies:", original_accuracy)
print("PCA Accuracies:", pca_accuracies)
print("UMAP Accuracies:", umap_accuracies)

#printeamos una breve conclusion
print("\nConclusion:")
print("-----------")
best_pca_accuracy = max(pca_accuracies)
best_pca_index = pca_accuracies.index(best_pca_accuracy)
best_pca_components = n_components[best_pca_index]

best_umap_accuracy = max(umap_accuracies)
best_umap_index = umap_accuracies.index(best_umap_accuracy)
best_umap_components = n_components[best_umap_index]

