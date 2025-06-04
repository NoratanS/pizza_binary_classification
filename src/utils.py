import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import itertools


def get_model(type: str, img_height=128, img_width=128, class_count=2):
    input_tensor = tf.keras.Input(shape=(img_height, img_width, 3))
    model = None

    # Resnet50 learning from nothing
    if type == "baseline-Resnet50":
        base_model = tf.keras.applications.ResNet50(weights=None,
                                                    include_top=False,
                                                    input_tensor=input_tensor)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Resnet50 transfer learning with imagenet weights
    elif type == "resnet_transfer_frozen":
        base_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                    include_top=False,
                                                    input_tensor=input_tensor)

        # weight freeze
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Resnet50 transfer learning with imagenet weights, but tunable lol
    elif type == "resnet-transfer-tunable":
        base_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                    include_top=False,
                                                    input_tensor=input_tensor)

        # Unfreeze layers
        base_model.trainable = True

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    return model


def plot_confusion_matrix(cm, class_names, normalize=False):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
