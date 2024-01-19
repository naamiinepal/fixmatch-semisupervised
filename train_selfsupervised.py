import sys
import torch

from semilearn.models.model import EfficientNetB0
import torch.utils.data as data
from torch import optim
from torch import nn
from ignite.metrics import Accuracy, Loss

from semilearn.core.criterions.cross_entropy import ce_loss
from semilearn.datasets.isic_dataset import (
    get_dataset,
    n_classes,
)
from semilearn.datasets.augmentations.transforms import (
    get_image_strong_augment_transform,
    get_image_transform,
    post_augment_transform,
    pre_augment_transform,
)


# Semi-Supervised Learning
'''
Here, we assume that there are only NUM_LABELS training samples, and large number of unlabelled images such that
UNLABELLED_SAMPLES >> LABELLED_SAMPLES.

This is quite common in real-world where annotation is expensive but unlabelled data are relatively available in abundance.

We work through code example (FixMatch Algorithm, in this case) to leverage unlabelled data for better representation learning.

The basic idea here is to 

i) train the model using limited labelled data
ii) as the model is getting trained, use the trained model to obtain (pseudo)labels for which the model is relatively certain (P_CUTOFF)
iii) and then obtain a aggressively modified version of the image and train the model to predict the above mentioned (in ii) 
In actual implementation 
'''


# training settings
NUM_EPOCHS = 30
BATCH_SIZE = 128
lr = 0.001

# dataset settings
IMG_SIZE = 224
NUM_LABELS = 100 # number of labelled data
mu = 5 # ratio of unlabelled to labelled data in a single batch 
P_CUTOFF = 0.95 # softmax threshold cutoff for pseudo label 
lambda_u = 1.0

verbose = True

pre_transform = pre_augment_transform(IMG_SIZE)
post_transform = post_augment_transform()
strong_augmentation = get_image_strong_augment_transform()

# how many batches to wait before logging training status
log_interval = 100

criterion = nn.CrossEntropyLoss()

val_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EfficientNetB0(num_classes=n_classes).to(device)
train_dataset = get_dataset(0,NUM_LABELS,img_transform=get_image_transform(IMG_SIZE))
unlabelled_dataset = get_dataset(NUM_LABELS,-1,img_transform=pre_transform)
test_dataset = get_dataset(NUM_LABELS,-1,img_transform=get_image_transform(IMG_SIZE))

# encapsulate data into dataloader form
labeled_train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
# assume the labels do not exist for the unlabeled dataset
unlabeled_train_loader = data.DataLoader(
    dataset=unlabelled_dataset, batch_size=mu * BATCH_SIZE, shuffle=True
)

train_loader_at_eval = data.DataLoader(
    dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)

val_loader = data.DataLoader(
    dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)

optimizer = optim.Adam(model.parameters(), lr=lr)


print("Train ", len(train_dataset), "Val", len(test_dataset))
img_batch, label_batch = next(iter(labeled_train_loader))
print(f'Batch {img_batch.shape}, label {label_batch.shape}')
img_batch, label_batch = next(iter(unlabeled_train_loader))
print(f'Batch {len(img_batch)} label {label_batch.shape}')

# Model Training
for epoch in range(NUM_EPOCHS):
    for labeled_batch_data, unlabeled_batch_data in zip(
        labeled_train_loader, unlabeled_train_loader
    ):
        inputs, labels = labeled_batch_data[0].to(device), labeled_batch_data[1].to(device)
        optimizer.zero_grad()

        # supervised loss
        pred_logits = model(inputs)
        supervised_loss = criterion(pred_logits, labels)

        # self-supervised loss
        unlabelled_input, _ = unlabeled_batch_data[0], unlabeled_batch_data[1]
        print(unlabelled_input)
        print(type(unlabelled_input))
        # ignore labels
        strongly_augmented_input = post_augment_transform(strong_augmentation(unlabelled_input)).to(device)
        unsupervised_pred_logits = model(strongly_augmented_input)
        with torch.no_grad():
            weak_augmented_img = post_augment_transform(unlabelled_input).to(device)
            unlabelled_pred_logits = model(unlabelled_input)
            unlabelled_pred_proba = torch.softmax(
                unlabelled_pred_logits.detach(), dim=-1
            )

            # compute mask
            max_probs, _ = torch.max(unlabelled_pred_proba, dim=-1)
            mask = max_probs.ge(P_CUTOFF).to(max_probs.dtype)

            # generate unlabeled targets using pseudo label
            pseudo_label = torch.argmax(unlabelled_pred_proba, dim=-1)

        # compute consistency loss
        unsupervised_loss = ce_loss(unsupervised_pred_logits, pseudo_label, reduction="none")
        unsupervised_loss = (unsupervised_loss * mask).mean()

        #total loss
        total_loss = supervised_loss + lambda_u * unsupervised_loss


        if verbose:
            print(f'total loss: {total_loss:.2f} supervised loss: {supervised_loss:.2f} unsupervised loss: {unsupervised_loss:.2f}')
        
        sys.exit(1)