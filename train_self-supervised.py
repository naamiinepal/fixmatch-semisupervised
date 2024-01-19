from time import gmtime, strftime
import torch
from semilearn.core.utils import compute_proba, seed_everything

from semilearn.models.model import EfficientNetB0
import torch.utils.data as data
from torch import optim
from torch import nn
from ignite.metrics import Accuracy, Loss
from ignite.engine import Engine, Events
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.handlers.checkpoint import ModelCheckpoint

from semilearn.core.criterions.cross_entropy import ce_loss
from semilearn.datasets.isic_dataset import (
    get_dataset,
    n_classes,
)

# training settings
NUM_EPOCHS = 30
BATCH_SIZE = 16
lr = 0.001

# dataset settings
IMG_SIZE = 224
mu = 5  # ratio of unlabelled to labelled data in a single batch
P_CUTOFF = 0.95  # softmax threshold cutoff for pseudo label
lambda_u = 1.0


SEED = 98123  # for reproducibility

seed_everything(SEED)

# how many batches to wait before logging training status
log_interval = 10

criterion = nn.CrossEntropyLoss()

val_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EfficientNetB0(num_classes=n_classes).to(device)
train_supervised_only = get_dataset(train=True, dataset_type="supervised_only")
train_unlabelled = get_dataset(train=True, dataset_type="unlabelled")
test_dataset = get_dataset(train=False)


# encapsulate data into dataloader form
# for reproducibility, seed the dataloader worker thread
g = torch.Generator()
g.manual_seed(SEED)
labeled_train_loader = data.DataLoader(
    dataset=train_supervised_only, batch_size=BATCH_SIZE, shuffle=True, generator=g
)
# assume the labels do not exist for the unlabeled dataset
unlabeled_train_loader = data.DataLoader(
    dataset=train_unlabelled, batch_size=mu * BATCH_SIZE, shuffle=True
)

train_loader_at_eval = data.DataLoader(
    dataset=train_supervised_only, batch_size=2 * BATCH_SIZE, shuffle=False
)

val_loader = data.DataLoader(
    dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)

optimizer = optim.Adam(model.parameters(), lr=lr)


print(
    "Train ",
    len(train_supervised_only),
    "Unlabelled",
    len(train_unlabelled),
    "Val",
    len(test_dataset),
)
img_batch, label_batch = next(iter(labeled_train_loader))
weak_img_batch, strong_img_batch, label_batch = next(iter(unlabeled_train_loader))


# Model Training
def train_step(labeled_batch_data, unlabeled_batch_data):
    model.train()
    inputs, labels = labeled_batch_data[0].to(device), labeled_batch_data[1].to(device)
    optimizer.zero_grad()

    # supervised loss
    pred_logits = model(inputs)
    supervised_loss = criterion(pred_logits, labels)

    # ignore labels
    weak_augmented_img, strongly_augmented_input = unlabeled_batch_data[0].to(
        device
    ), unlabeled_batch_data[1].to(device)

    unsupervised_pred_logits = model(strongly_augmented_input)

    # pseudo label
    with torch.no_grad():
        unlabelled_pred_logits = model(weak_augmented_img)
        unlabelled_pred_proba = compute_proba(unlabelled_pred_logits.detach())

        # Confidence score
        max_probs, _ = torch.max(unlabelled_pred_proba, dim=-1)
        mask = max_probs.ge(P_CUTOFF).to(max_probs.dtype)

        # generate hard unlabeled targets using pseudo label
        pseudo_label = torch.argmax(unlabelled_pred_proba, dim=-1)

    # compute consistency loss
    unsupervised_loss = ce_loss(
        unsupervised_pred_logits, pseudo_label, reduction="none"
    )
    # choose only those pseudo labels with high confidence score
    unsupervised_loss = (unsupervised_loss * mask).mean()

    # total loss
    total_loss = supervised_loss + lambda_u * unsupervised_loss

    total_loss.backward()  # calculate gradients
    optimizer.step()  # update weights

    return {
        "batch_total_loss": total_loss.item(),
        "batch_supervised_loss": supervised_loss.item(),
        "batch_unsupervised_loss": unsupervised_loss.item(),
        "batch_confident_pseudo_labels": mask.sum().item(),
    }


def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        return y_pred, y


def log_training_loss(train_out_dict, state):
    print(
        f"Epoch[{state['epoch']}], Iter[{state['iteration']}] Loss: {train_out_dict['batch_total_loss']:.2f}"
    )


train_evaluator = Engine(validation_step)
val_evaluator = Engine(validation_step)

# attach metrics to evaluators
for name, metric in val_metrics.items():
    metric.attach(train_evaluator, name)

for name, metric in val_metrics.items():
    metric.attach(val_evaluator, name)


def log_training_results():
    train_evaluator.run(train_loader_at_eval)
    metrics = train_evaluator.state.metrics
    print(
        f"Training Results - Epoch[{train_evaluator.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )


def log_validation_results():
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    print(
        f"Validation Results - Epoch[{val_evaluator.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )


date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

# Checkpoint to store n_saved best models wrt score function


# Score function to return current value of any metric we defined above in val_metrics
def score_function(engine):
    return engine.state.metrics["accuracy"]


model_checkpoint = ModelCheckpoint(
    f"checkpoint/semi-supervised/{date_time}",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
)

# Save the model after every epoch of val_evaluator is completed
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

# Define a Tensorboard logger
tb_logger = TensorboardLogger(log_dir=f"tb-logger/semi-supervised/{date_time}")

# Attach handler for plotting both evaluators' metrics after every epoch completes
for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
    )


for epoch in range(NUM_EPOCHS):
    i = 0
    for labeled_batch_data, unlabeled_batch_data in zip(
        labeled_train_loader, unlabeled_train_loader
    ):
        train_out_dict = train_step(labeled_batch_data, unlabeled_batch_data)

        if i % log_interval:
            tb_logger.writer.add_scalars(
                "training",
                train_out_dict,
                global_step=epoch * len(labeled_train_loader) + i,
            )
            log_training_loss(train_out_dict, {"epoch": epoch + 1, "iteration": i})
        i = i + 1  # iteration counter
    log_training_results()
    log_validation_results()
