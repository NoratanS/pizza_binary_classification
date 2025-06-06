from clearml import Task
from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange, UniformParameterRange

# Task init
task = Task.init(project_name="pizza_binary_classification", task_name="hpo_launcher_v2", task_type=Task.TaskTypes.optimizer)

# Base task
base_task = Task.get_task(project_name="pizza_binary_classification", task_name="hpo_train_task_v4")

# Configuration
optimizer = HyperParameterOptimizer(
    base_task_id=base_task.id,
    hyper_parameters=[
        DiscreteParameterRange("General/network", ["mobilenet", "resnet50", "vgg16", "resnet101", "inceptionv3"]),
        DiscreteParameterRange("General/img_size", [96, 128, 160, 224]),
        DiscreteParameterRange("General/use_pretrained", [True, False]),
        DiscreteParameterRange("General/freeze_base", [True, False]),
        DiscreteParameterRange("General/use_dropout", [True, False]),
        UniformParameterRange("General/dropout_rate", 0.0, 0.5),
        DiscreteParameterRange("General/batch_size", [16, 32, 64]),
    ],
    objective_metric_title="accuracy",
    objective_metric_series="val",
    objective_metric_sign="max",
    execution_queue="default",
    max_number_of_concurrent_tasks=2,
    total_max_jobs=20,
    min_iteration_per_job=3,
    optimizer_type="random_search"
)

print("HPO will optimize the following parameters:")
for p in optimizer.hyper_parameters:
    print(f" - {p.name}")

optimizer.set_report_period(60)
optimizer.start()
optimizer.wait()
