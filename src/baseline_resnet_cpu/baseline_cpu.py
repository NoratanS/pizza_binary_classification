import tensorflow as tf
import matplotlib.pyplot as plt
import time

from clearml import Task, Dataset
from src.utils import get_model, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from PIL import Image

task = Task.init(project_name='pizza_binary_classification', task_name='baseline-Resnet50', output_uri=True)
logger = task.get_logger()

data = Dataset.get(dataset_id='c653d71c5fc64dc4965e09c442b2bae3').get_local_copy()

params = {
    'batch_size': 16,
    'img_height': 128,
    'img_width': 128,
    'epochs': 20
}
task.connect(params)

# Data split
train_ds = tf.keras.utils.image_dataset_from_directory(
    data,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(params['img_height'], params['img_width']),
    batch_size=params['batch_size']
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(params['img_height'], params['img_width']),
    batch_size=params['batch_size']
)

class_names = train_ds.class_names
class_count = len(class_names)
print(class_names)

# Choosing a model
model = get_model('baseline-Resnet50',
                  img_height=params['img_height'],
                  img_width=params['img_width'],
                  class_count=class_count)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training
start = time.time()
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=params['epochs']
)
training_time = time.time() - start

# Clearml logs
for epoch in range(params['epochs']):
    train_loss = history.history['loss'][epoch]
    val_loss = history.history['val_loss'][epoch]
    train_acc = history.history['accuracy'][epoch]
    val_acc = history.history['val_accuracy'][epoch]

    logger.report_scalar("loss", "train", value=train_loss, iteration=epoch + 1)
    logger.report_scalar("loss", "val",   value=val_loss,   iteration=epoch + 1)
    logger.report_scalar("accuracy", "train", value=train_acc, iteration=epoch + 1)
    logger.report_scalar("accuracy", "val",   value=val_acc,   iteration=epoch + 1)

logger.report_text(f"Total training time: {training_time:.2f} seconds")

val_loss, val_accuracy = model.evaluate(val_ds)

# confusion matrix
y_true, y_pred = [], []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype("int32").flatten())


cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, class_names)

# Save cm image and send to clearml
plt.savefig("baseline_cpu_confusion_matrix.png")
image = Image.open("baseline_cpu_confusion_matrix.png")
logger.report_image("Confusion Matrix", "val", iteration=0, image=image)
plt.close()


# show results
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")

print(f"\nTotal training time: {training_time:.2f} seconds")
