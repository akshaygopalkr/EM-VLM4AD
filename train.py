from transformers import T5Tokenizer, TrainingArguments, Trainer
from torchvision import transforms
import json
import os
import time
from torch.utils.data import DataLoader
import torch
import argparse
from modules.multi_frame_dataset import MultiFrameDataset
from modules.multi_frame_model import print_trainable_parameters, DriveVLMT5
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model(model, model_name):
    # Save the model into the designated folder
    path = os.path.join('multi_frame_results', timestr, model_name + '.pth')
    torch.save(model, path)


def val_model(dloader, val_model):
    val_model.eval()
    val_loss = 0

    for idx, (inputs, imgs, labels) in tqdm(enumerate(dloader), total=len(dloader)):
        outputs = val_model(inputs, imgs, labels)
        val_loss += outputs.loss.item()

    return val_loss / len(val_dataloader)


def save_stats(train_loss, val_loss, epochs, lr):
    stats_dict = {
        'losses': losses,
        'val losses': val_losses,
        'min train loss': train_loss,
        'min val loss': val_loss,
        'epochs': epochs,
        'learning rate': lr,
        'LM': 'T5-Base',
        'Image Embedding': 'Patch'
    }

    # Save stats into checkpoint
    with open(os.path.join('multi_frame_results', timestr, 'stats.json'), 'w') as f:
        json.dump(stats_dict, f)


def plot_loss(training_loss, val_loss):
    num_epochs = len(training_loss)

    plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Num epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('multi_frame_results', timestr, 'loss.png'))


def custom_train(train_loss, val_loss, best_model, epochs, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

    for epoch in range(epochs, config.epochs):
        print('-------------------- EPOCH ' + str(epoch) + ' ---------------------')
        model.train()
        epoch_loss = 0

        for step, (inputs, imgs, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            # print(inputs.shape, imgs.shape, labels.shape)

            # Forward pass through model
            outputs = model(inputs, imgs, labels)

            # Calculate loss
            loss = outputs.loss
            epoch_loss += loss.item()

            if step % config.checkpoint_frequency == 0:
                print()
                print('Loss: ' + str(loss.item()))

                # Get the hidden states (output)
                hidden_states = outputs.logits

                # Perform decoding (e.g., greedy decoding)
                outputs = torch.argmax(hidden_states, dim=-1)

                text_outputs = [processor.decode(output.to('cpu'), skip_special_tokens=True) for output in outputs]
                text_questions = [processor.decode(q.to('cpu'), skip_special_tokens=True) for q in inputs]
                text_labels = [processor.decode(a.to('cpu'), skip_special_tokens=True) for a in labels]
                print()
                print('Questions:')
                print(text_questions)
                print()
                print('Generated Answers:')
                print(text_outputs)
                print()
                print('Ground Truth Answers:')
                print(text_labels)

            # Back-propogate
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Get train and val loss per batch
        epoch_train_loss = epoch_loss / len(train_dataloader)
        losses.append(epoch_train_loss)

        epoch_val_loss = val_model(val_dataloader, model)
        val_losses.append(epoch_val_loss)

        if not val_loss or min(epoch_val_loss, val_loss) == epoch_val_loss:
            val_loss = epoch_val_loss
            best_model = deepcopy(model.state_dict())
        if not train_loss or min(train_loss, epoch_train_loss) == epoch_train_loss:
            train_loss = epoch_train_loss

        # Adjust learning rate scheduler
        scheduler.step()

        print('Training Loss: ' + str(epoch_train_loss))
        print('Validation Loss: ' + str(epoch_val_loss))
        print('---------------------------------------------')

        # Save model and stats for checkpoints
        save_model(best_model, 'latest_model')
        epochs += 1
        save_stats(train_loss, val_loss, epochs, scheduler.get_last_lr()[0])

    # Save the model and plot the loss
    plot_loss(losses, val_losses)
    return train_loss, val_loss


def train():
    training_config = TrainingArguments(
        output_dir="agopalkr/EfficientDriveLM",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        config=training_config,
        train_dataset=train_dset,
        eval_dataset=val_dset,
    )

    trainer.train()
    model.push_to_hub("agopalkr/EfficientDriveLM")


def save_experiment(statistics):
    """
    Saves the experiment multi_frame_results to a csv
    :param config: The hyperparameters used
    :param statistics: The accuracies for the training, validation, and test sets
    """
    trial_dict = {
        'Model name': [timestr],
        'Learning rate': [config.learning_rate],
        'Weight decay': [config.weight_decay],
        'Batch size': [config.batch_size],
        'Epochs': [config.epochs],
        'LoRA finetuning': [config.lora],
        'GPA Hidden Size': [config.gpa_hidden_size],
        'LoRA Dimension': [config.lora_dim],
        'LoRA Alpha': [config.lora_alpha],
        'LoRA Dropout': [config.lora_dropout],
        'Freeze T5': [config.freeze_lm],
        'Min Training Loss': [statistics[0]],
        'Min Validation Loss': [statistics[1]],
        'Min Testing Loss': [statistics[2]],
    }

    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(os.path.join('multi_frame_results', timestr, 'multi_frame_results.csv'), index=False, header=True)


def params():

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", default=1e-4, type=float,
                        help="Model learning rate starting point, default is 1e-4.")
    parser.add_argument("--batch-size", default=4, type=int,
                        help="Batch size per GPU/CPU for training and evaluation, defaults to 4.")
    parser.add_argument("--weight-decay", default=0.05, type=float,
                        help="L2 Regularization, default is 0.05")
    parser.add_argument("--epochs", default=15, type=int,
                        help="Number of epochs to train for, default is 15")
    parser.add_argument("--hf-train", action='store_true',
                        help="Whether to use HuggingFace default training or custom training loop")
    parser.add_argument('--gpa-hidden-size', default=128, type=int, help='Hidden dimension for Gated Pooling Attention, '
                                                                         'default is 128')
    parser.add_argument('--freeze-lm', action='store_true', help='Freeze LM during training')
    parser.add_argument('--lm', default='T5-Base', choices=['T5-Base', 'T5-Large'], type=str, help='Backbone LM to use, '
                                                                                        'use \'T5-Base\' for T5-Medium')
    parser.add_argument('--checkpoint-frequency', default=10000, type=int, help='Frequency of showing example outputs')
    parser.add_argument('--lora', action='store_true', help='Perform LoRA finetuning, recommend if '
                                                            'using T5-Large backbone LM')
    parser.add_argument('--lora-dim', default=16, type=int, help='LoRA dimension')
    parser.add_argument('--lora-alpha', default=32, type=int, help='LoRA alpha')
    parser.add_argument('--lora-dropout', default=0.05, type=float, help='LoRA dropout')
    parser.add_argument('--num-workers', default=0, type=int, help='# of Workers used by Dataloader')
    parser.add_argument('--load-checkpoint', action='store_true', help='Whether to load a checkpoint from '
                                                                       'multi_frame_results folder')
    parser.add_argument('--checkpoint-file', default='T5-Medium', type=str, help='The checkpoint to load from '
                                                                                 'multi_frame_results directory')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    timestr = time.strftime("%Y%m%d-%H%M%S")

    config = params()

    losses = []
    val_losses = []
    min_train_loss = None
    min_val_loss = None
    best_model = None
    epochs_ran = 0

    # Load processors and models
    model = DriveVLMT5(config)
    model.to(device)
    print('Trainable Parameters for full model')
    print_trainable_parameters(model)

    if config.lm == 'T5-Base':
        processor = T5Tokenizer.from_pretrained('google-t5/t5-base')
    else:
        processor = T5Tokenizer.from_pretrained('google-t5/t5-large')

    train_dset = MultiFrameDataset(
        input_file=os.path.join('data', 'multi_frame',
                                'multi_frame_train.json'),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )
    val_dset = MultiFrameDataset(
        input_file=os.path.join('data', 'multi_frame',
                                'multi_frame_val.json'),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )
    test_dset = MultiFrameDataset(
        input_file=os.path.join('data', 'multi_frame',
                                'multi_frame_test.json'),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )

    # Create Dataloaders
    train_dataloader = DataLoader(train_dset, shuffle=True, batch_size=config.batch_size,
                                  num_workers=config.num_workers, collate_fn=train_dset.collate_fn)
    val_dataloader = DataLoader(val_dset, shuffle=True, batch_size=config.batch_size,
                                num_workers=config.num_workers, collate_fn=train_dset.collate_fn)
    test_dataloader = DataLoader(test_dset, shuffle=True, batch_size=config.batch_size,
                                 num_workers=config.num_workers, collate_fn=train_dset.collate_fn)

    if not config.hf_train:

        # Load checkpoint if neccesary:
        if config.load_checkpoint:

            print('Loading model from ' + config.checkpoint_file)

            # Load the model and stats from the checkpoint
            model.load_state_dict(torch.load(os.path.join('multi_frame_results', config.checkpoint_file,
                                                          'latest_model.pth')))
            best_model = DriveVLMT5(config)
            best_model.load_state_dict(torch.load(os.path.join('multi_frame_results', config.checkpoint_file,
                                                               'latest_model.pth')))

            with open(os.path.join('multi_frame_results', config.checkpoint_file, 'stats.json'), 'r') as f:
                stats = json.load(f)

            min_train_loss, min_val_loss, losses, val_losses, epochs_ran = stats['min train loss'], stats[
                'min val loss'], stats['losses'], stats['val losses'], stats['epochs']

            print(f'Minimum Training Loss: {min_train_loss}')
            print(f'Training Losses: {losses}')
            print(f'Minimum Validation Loss: {min_val_loss}')
            print(f'Validation Losses: {val_losses}')
            print(f'Epochs ran: {epochs_ran}')
            timestr = config.checkpoint_file
        else:
            checkpoint_path = os.path.join('multi_frame_results', timestr)
            print(f'All model checkpoints and training stats will be saved in {checkpoint_path}')
            os.mkdir(os.path.join('multi_frame_results', timestr))

        # If loading a checkpoint, use the learning rate from the last epoch
        if config.load_checkpoint:
            lr = stats['learning rate']
        else:
            lr = config.learning_rate

        min_train_loss, min_val_loss = custom_train(min_train_loss, min_val_loss, best_model, epochs_ran, lr)
        best_model = DriveVLMT5(config)
        best_model.load_state_dict(torch.load(os.path.join('multi_frame_results', timestr, 'latest_model.pth')))
        best_model.to(device)
        test_loss = val_model(test_dataloader, best_model)
        statistics = [min_train_loss, min_val_loss, test_loss]
        save_experiment(statistics)
    else:
        train()
