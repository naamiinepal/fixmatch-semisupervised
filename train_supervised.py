from time import gmtime, strftime

from ignite.engine import Engine, Events
from ignite.handlers.checkpoint import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

import torch

from ssl_fork.models.model import BasicNet
from ssl_fork.datasets.medmnist_dataset import n_channels,n_classes
from ssl_fork.datasets.medmnist_dataset import train_dataset, test_dataset
import torch.utils.data as data
from torch import optim
from torch import nn
from ignite.metrics import Accuracy,Loss

NUM_EPOCHS = 30
BATCH_SIZE = 128
lr = 0.001

# how many batches to wait before logging training status
log_interval = 100

criterion = nn.CrossEntropyLoss()

val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BasicNet(num_classes=n_classes,in_channels=n_channels,dropout=0.2).to(device)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
val_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

batch = next(iter(train_loader))
optimizer = optim.Adam(model.parameters(), lr=lr)


def supervised_train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch[0].to(device), batch[1].to(device)
    y_pred = model(x)
    loss = criterion(y_pred,y.squeeze(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x,y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        return y_pred, y.squeeze(-1)

trainer = Engine(supervised_train_step)

train_evaluator = Engine(validation_step)
val_evaluator = Engine(validation_step)



# attach metrics to evaluators
for name, metric in val_metrics.items():
    metric.attach(train_evaluator,name)

for name, metric in val_metrics.items():
    metric.attach(val_evaluator, name)


@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

# Score function to return current value of any metric we defined above in val_metrics
def score_function(engine):
    return engine.state.metrics["accuracy"]

date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

# Checkpoint to store n_saved best models wrt score function
model_checkpoint = ModelCheckpoint(
    f"checkpoint/{date_time}",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer), # helps fetch the trainer's state
)
  
# Save the model after every epoch of val_evaluator is completed
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

# Define a Tensorboard logger
tb_logger = TensorboardLogger(log_dir=f"tb-logger/{date_time}")

# Attach handler to plot trainer's loss every 100 iterations
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

# Attach handler for plotting both evaluators' metrics after every epoch completes
for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )


trainer.run(train_loader, max_epochs=NUM_EPOCHS)

tb_logger.close()