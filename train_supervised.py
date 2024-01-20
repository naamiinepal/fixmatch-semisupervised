from time import gmtime, strftime
from ignite.engine import Engine, Events
from ignite.handlers.checkpoint import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

import torch

import torch.utils.data as data
from torch import optim
from torch import nn
from ignite.metrics import Accuracy, Loss, Precision, Recall,Fbeta

from semilearn.core.utils import get_latest_checkpoint, seed_everything

from semilearn.models.model import EfficientNetB0
from semilearn.datasets.augmentations.transforms import get_image_transform
from semilearn.datasets.isic_dataset import get_test_dataset, get_train_dataset, get_val_dataset
from argparse import ArgumentParser

SEED = 98123  # for reproducibility





seed_everything(SEED)  # additionally seed the torch generator

parser = ArgumentParser()
parser.add_argument('--supervised_only','-so',default=False,action='store_true')
parser.add_argument('--num_epochs',default=30,type=int)
args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = 16
lr = 0.001
IMG_SIZE = 224

# how many batches to wait before logging training status
log_interval = 10

criterion = nn.CrossEntropyLoss()

val_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion),'f1score':Fbeta(beta=1.0)}
test_metrics = {'accuracy':Accuracy(), 'f1score':Fbeta(beta=1.0)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EfficientNetB0(num_classes=2, load_imagenet_weights=True).to(device)

# encapsulate data into dataloader form
img_transform = get_image_transform(IMG_SIZE)

if args.supervised_only:
    dataset_type = 'supervised_only'
else:
    dataset_type = 'fully_supervised'

train_dataset = get_train_dataset(img_transform=img_transform,dataset_type=dataset_type)
val_dataset = get_val_dataset(img_transform=img_transform)
test_dataset = get_test_dataset(img_transform=img_transform)

# for reproducibility, seed the dataloader worker thread
g = torch.Generator()
g.manual_seed(SEED)
train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g
)
train_loader_at_eval = data.DataLoader(
    dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)
val_loader = data.DataLoader(
    dataset=val_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)
test_loader = data.DataLoader(dataset=test_dataset,batch_size=2*BATCH_SIZE,shuffle=False)

batch = next(iter(train_loader))
optimizer = optim.Adam(model.parameters(), lr=lr)


def supervised_train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch[0].to(device), batch[1].to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y.squeeze(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        return y_pred, y.squeeze(-1)


trainer = Engine(supervised_train_step)

train_evaluator = Engine(validation_step)
val_evaluator = Engine(validation_step)
test_evaluator = Engine(validation_step)

# attach metrics to evaluators
for name, metric in val_metrics.items():
    metric.attach(train_evaluator, name)

for name, metric in val_metrics.items():
    metric.attach(val_evaluator, name)

for name, metric in test_metrics.items():
    metric.attach(test_evaluator, name)

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(
        f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}"
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(
        f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    print(
        f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )


# Score function to return current value of any metric we defined above in val_metrics
def score_function(engine):
    return engine.state.metrics["f1score"]


date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

# Checkpoint to store n_saved best models wrt score function
model_checkpoint = ModelCheckpoint(
    f"checkpoint/{dataset_type}/{date_time}",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="f1score",
    global_step_transform=global_step_from_engine(
        trainer
    ),  # helps fetch the trainer's state
)

# Save the model after every epoch of val_evaluator is completed
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

# Define a Tensorboard logger
tb_logger = TensorboardLogger(log_dir=f"tb-logger/{dataset_type}/{date_time}")

# Attach handler to plot trainer's loss every 100 iterations
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=log_interval),
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

# Load best validation model and report test accuracy
ckpt_path = get_latest_checkpoint(f'checkpoint/{dataset_type}')
checkpoint_dict = torch.load(ckpt_path, map_location=device) 
model.load_state_dict(checkpoint_dict)
test_evaluator.run(test_loader)
metrics = test_evaluator.state.metrics
for key,val in metrics.items():
    tb_logger.writer.add_scalar(f'test/{key}',val)

tb_logger.close()
