import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import itertools

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),   # was 0.1
    tf.keras.layers.RandomZoom(0.05),       # was 0.1
    # tf.keras.layers.RandomContrast(0.1),  # turned off
], name="data_augmentation")


def get_model(model_name: str,
              img_height=128,
              img_width=128,
              class_count=2,
              use_pretrained=False,
              freeze_base=True,
              use_dropout=False,
              dropout_rate=0.3,
              use_augmentation=False):
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

    # Available models
    model_map = {
        "resnet50": tf.keras.applications.ResNet50,
        "resnet101": tf.keras.applications.ResNet101,
        "vgg16": tf.keras.applications.VGG16,
        "inceptionv3": tf.keras.applications.InceptionV3,
        "mobilenet": tf.keras.applications.MobileNet
    }

    if model_name not in model_map:
        raise ValueError(f"Nieznany model: {model_name}")

    BaseModel = model_map[model_name]

    # Input
    input_tensor = Input(shape=(img_height, img_width, 3))

    # Augmentation (if active)
    x = input_tensor
    if use_augmentation:
        data_augmentation = Sequential([
            RandomFlip("horizontal"),
            RandomRotation(0.05),
            RandomZoom(0.05)
        ], name="data_augmentation")
        x = data_augmentation(x)

    # Init model
    weights = 'imagenet' if use_pretrained else None
    base_model = BaseModel(weights=weights, include_top=False, input_tensor=x)

    # Freeze weights (if TL and freeze=True)
    if use_pretrained and freeze_base:
        base_model.trainable = False
    else:
        base_model.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Dropout
    if use_dropout:
        x = Dropout(dropout_rate)(x)

    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)  # binary classification

    model = Model(inputs=input_tensor, outputs=output)
    return model


def get_model_legacy(type: str, img_height=128, img_width=128, class_count=2):
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

        model = tf.keras.Model(inputs=input_tensor, outputs=output)

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

    elif type == "resnet-baseline-augmentation":
        x = data_augmentation(input_tensor)
        base_model = tf.keras.applications.ResNet50(weights=None,
                                                    include_top=False,
                                                    input_tensor=x)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=input_tensor, outputs=output)

    elif type == "resnet-baseline-dropout":
        base_model = tf.keras.applications.ResNet50(weights=None,
                                                    include_top=False,
                                                    input_tensor=input_tensor)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=input_tensor, outputs=output)

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
