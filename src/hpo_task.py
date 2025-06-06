import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from PIL import Image

from clearml import Task, Dataset
from src.utils import get_model, plot_confusion_matrix

# Task init
task = Task.init(project_name='pizza_binary_classification', task_name='hpo_train_task_v2', output_uri=True)
task.set_task_type(Task.TaskTypes.training)
task.set_script(__file__)

logger = task.get_logger()

# Hiperparameters
params = {
    'network': 'mobilenet',
    'img_size': 128,
    'batch_size': 16,
    'epochs': 20,
    'use_pretrained': True,
    'freeze_base': True,
    'use_dropout': False,
    'dropout_rate': 0.0,
    'use_augmentation': False,
}
task.connect(params)

# Dataset
data = Dataset.get(dataset_id='c653d71c5fc64dc4965e09c442b2bae3').get_local_copy()

train_ds = tf.keras.utils.image_dataset_from_directory(
    data,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(params['img_size'], params['img_size']),
    batch_size=params['batch_size']
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(params['img_size'], params['img_size']),
    batch_size=params['batch_size']
)

class_names = train_ds.class_names
class_count = len(class_names)

# Preprocessing
preprocessing_map = {
    'mobilenet': tf.keras.applications.mobilenet.preprocess_input,
    'resnet50': tf.keras.applications.resnet50.preprocess_input,
    'vgg16': tf.keras.applications.vgg16.preprocess_input,
    'resnet101': tf.keras.applications.resnet.preprocess_input,
    'inceptionv3': tf.keras.applications.inception_v3.preprocess_input,
}

preprocess_input = preprocessing_map.get(params['network'])

if preprocess_input is None:
    raise ValueError(f"No preprocess_input for specified network: {params['network']}")

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Model
model = get_model(
    model_name=params['network'],
    img_height=params['img_size'],
    img_width=params['img_size'],
    class_count=class_count,
    use_pretrained=params['use_pretrained'],
    freeze_base=params['freeze_base'],
    use_dropout=params['use_dropout'],
    dropout_rate=params['dropout_rate'],
    use_augmentation=params['use_augmentation']
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

start = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=params['epochs'])
training_time = time.time() - start

# Metric Log
for epoch in range(params['epochs']):
    logger.report_scalar("loss", "train", history.history['loss'][epoch], epoch + 1)
    logger.report_scalar("loss", "val", history.history['val_loss'][epoch], epoch + 1)
    logger.report_scalar("accuracy", "train", history.history['accuracy'][epoch], epoch + 1)
    logger.report_scalar("accuracy", "val", history.history['val_accuracy'][epoch], epoch + 1)

# Evaluate
val_loss, val_accuracy = model.evaluate(val_ds)

# F1-score and confusion matrix
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype("int32").flatten())

cm = confusion_matrix(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
logger.report_scalar("f1_score", "val", f1, params['epochs'])

plot_confusion_matrix(cm, class_names)

title = (
    f"{params['network']}_size-{params['img_size']}_"
    f"pretrained-{params['use_pretrained']}_dropout-{str(params['dropout_rate']).replace('.', '')}"
)
plt.savefig(f"{title}.png")
img = Image.open(f"{title}.png").convert("RGB")
logger.report_image(f"{title}", "val", iteration=0, image=img)
plt.close()

print(f"Final Accuracy: {val_accuracy:.3f} | F1 Score: {f1:.3f} | Time: {training_time:.2f} sec")
