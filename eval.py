import argparse
import pdb

import torch
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
from modules.multi_frame_dataset import MultiFrameDataset
from modules.multi_frame_model import DriveVLMT5
from tqdm import tqdm as progress_bar
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def val_model(dloader):
    model.eval()
    ids_answered = set()
    test_data = []

    with torch.no_grad():
        for idx, (q_texts, encodings, imgs, labels, img_paths) in progress_bar(enumerate(dloader), total=len(dloader)):

            # Get the hidden states (output)
            hidden_states = model(encodings, imgs, labels).logits

            # Perform decoding (e.g., greedy decoding)
            outputs = torch.argmax(hidden_states, dim=-1)

            # Get the text output
            text_outputs = [processor.decode(output, skip_special_tokens=True) for output in outputs]

            if idx % 100 == 0:
                print(q_texts)
                print(text_outputs)

            for image_path, q_text, text_output in zip(img_paths, q_texts, text_outputs):

                img_key = image_path[0]

                # Skip duplicate questions
                if image_id_dict[img_key + ' ' + q_text][0] in ids_answered:
                    continue
                if len(text_output) > config.max_len:
                    continue

                ids_answered.add(image_id_dict[img_key + ' ' + q_text][0])
                test_data.append({'image_id': image_id_dict[img_key + ' ' + q_text][0], 'caption': text_output})

    # Save test output to file
    with open(os.path.join('multi_frame_results', config.model_name, 'predictions.json'), 'w') as f:
        json.dump(test_data, f)


def save_experiment():
    """
    Saves the experiment results to a csv
    :param config: The hyperparameters used
    :param statistics: The accuracies for the training, validation, and test sets
    """

    trial_dict = {}

    # Add metrics to dictionary
    for metric, score in coco_eval.eval.items():
        trial_dict[metric] = [score]

    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(
        os.path.join('multi_frame_results', config.model_name, 'metrics.csv'),
        index=False, header=True)

def params():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=4, type=int,
                        help="Batch size per GPU/CPU for training and evaluation, defaults to 4.")
    parser.add_argument("--epochs", default=15, type=int,
                        help="Number of epochs to train for, default is 15")
    parser.add_argument('--gpa-hidden-size', default=128, type=int, help='Hidden dimension for Gated Pooling Attention, '
                                                                         'default is 128')
    parser.add_argument('--freeze-lm', action='store_true', help='Freeze LM during training')
    parser.add_argument('--lm', default='T5-Base', choices=['T5-Base', 'T5-Large'], type=str, help='Backbone LM to use, '
                                                                                        'use \'T5-Base\' for T5-Medium')
    parser.add_argument('--lora', action='store_true', help='Perform LoRA finetuning, recommend if '
                                                            'using T5-Large backbone LM')
    parser.add_argument('--lora-dim', default=64, type=int, help='LoRA dimension')
    parser.add_argument('--lora-alpha', default=32, type=int, help='LoRA alpha')
    parser.add_argument('--lora-dropout', default=0.05, type=float, help='LoRA dropout')
    parser.add_argument('--max-len', default=512, type=int, help='Max length for generating sequence')
    parser.add_argument('--num-workers', default=0, type=int, help='# of Workers used by Dataloader')
    parser.add_argument('--model-name', default='T5-Medium', type=str, help='The checkpoint to load from '
                                                                                 'multi_frame_results directory')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    config = params()

    # Load processors and models
    model = DriveVLMT5(config)
    model.to(device)

    if config.lm == 'T5-Base':
        processor = T5Tokenizer.from_pretrained('google-t5/t5-base')
    else:
        processor = T5Tokenizer.from_pretrained('google-t5/t5-large')

    processor.add_tokens('<')

    model.load_state_dict(
        torch.load(os.path.join('multi_frame_results', config.model_name,
                                'latest_model.pth')))

    # Load dataset and dataloader
    test_dset = MultiFrameDataset(
        input_file=os.path.join('data', 'multi_frame',
                                'multi_frame_test.json'),
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )
    test_dloader = DataLoader(test_dset, shuffle=True, batch_size=config.batch_size, drop_last=True,
                              collate_fn=test_dset.test_collate_fn)

    # Load in image ids
    with open(os.path.join('data', 'multi_frame', 'image_id.json')) as f:
        image_id_dict = json.load(f)

    # Get the loss and predictions from the model
    val_model(test_dloader)

    annotation_file = os.path.join('data', 'multi_frame', 'multi_frame_test_coco.json')
    results_file = os.path.join('multi_frame_results', config.model_name,
                                'predictions.json')

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # Save the experiment results
    save_experiment()
