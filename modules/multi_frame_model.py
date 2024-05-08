from transformers import T5ForConditionalGeneration
from torchvision.models import vit_b_32
import torch.nn as nn
import torch
from peft import LoraConfig, get_peft_model, LoftQConfig

VIT_HIDDEN_STATE = 768
VIT_SEQ_LENGTH = 49

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class DriveVLMT5(nn.Module):

    def __init__(self, config):

        super().__init__()

        # Make tokenizer and text model
        if config.lm == 'T5-Base':
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base')
        else:
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-large')

            # For quantization
            loftq_config = LoftQConfig(loftq_bits=8)

            # Create LoRA model
            lora_config = LoraConfig(
                r=config.lora_dim,
                lora_alpha=config.lora_alpha,
                loftq_config=loftq_config,
                lora_dropout=config.lora_dropout,
                bias='none',
                target_modules=['q', 'v']
            )
            self.model = get_peft_model(self.model, lora_config)

        hidden_size = self.model.config.d_model

        print('Trainable Parameters for LM model:')
        print_trainable_parameters(self.model)

        # Create instance for multi-view processor
        self.mvp = self.MultiViewProcessor(config.gpa_hidden_size, hidden_size, config.lm, freeze=True)

    class MultiViewProcessor(nn.Module):

        def __init__(self, gpa_hidden_size, hidden_size, lm, freeze=False):

            super().__init__()

            # Use ViT for image embeddings
            self.img_model = vit_b_32(weights='DEFAULT')
            self.lm = lm

            # Modal embedding to distinguish between image and text
            self.modal_embeddings = nn.Embedding(2, hidden_size)
            self.modal_embeddings.weight.data.normal_(mean=0.0, std=0.02)

            # If we are freezing the CLIP embeddings
            if freeze:
                for param in self.img_model.parameters():
                    param.requires_grad = False

            # Set matrices based on MIVC paper
            self.w = nn.Linear(in_features=gpa_hidden_size, out_features=1)
            self.Z = nn.Sequential(
                nn.Linear(in_features=VIT_HIDDEN_STATE * VIT_SEQ_LENGTH, out_features=gpa_hidden_size, bias=False),
                nn.Tanh()
            )
            self.G = nn.Sequential(
                nn.Linear(in_features=VIT_HIDDEN_STATE * VIT_SEQ_LENGTH, out_features=gpa_hidden_size, bias=False),
                nn.Sigmoid()
            )

            if self.lm != 'T5-Base':
                self.img_projection_layer = nn.Linear(in_features=VIT_HIDDEN_STATE, out_features=hidden_size)

        def gpa(self, img_embeddings):

            """"
            Calculates the gated-pooling attention score for the image embeddings
            :param img_embeddings: (6x768) dimensional
            :return single embedding of size (768,)
            """

            # Get weights for gated pooling attention
            gpa_weights = torch.softmax(self.w(self.Z(img_embeddings) * self.G(img_embeddings)), dim=0)

            # Take a linear combination of all the image embeddings
            fused_embeddings = torch.sum(gpa_weights * img_embeddings, dim=0)

            return fused_embeddings

        def get_img_embedding(self, imgs):

            N = imgs.shape[0]

            # Process into patches (N x 6 x 49 x H)
            merged_embedding = torch.stack([self.img_model._process_input(img) for img in imgs], dim=0)

            # Concatenate the batch class tokens -> (N, 6, 50, H)
            batch_class_tokens = self.img_model.class_token.expand(merged_embedding.shape[1], -1, -1).repeat(N, 1, 1, 1)
            merged_embedding = torch.cat([batch_class_tokens, merged_embedding], dim=2)

            # Add positional embeddings and remove class token -> (N, 6, 49, H)
            merged_embedding += self.img_model.encoder.pos_embedding.repeat(N, 1, 1, 1)
            merged_embedding = merged_embedding[:, :, 1:]

            # Get merged embedding and reshape to 2D embedding -> (N, 1, 49, H)
            merged_embedding = torch.stack([self.gpa(embedding.flatten(start_dim=1)).reshape(VIT_SEQ_LENGTH,
                                                                                             VIT_HIDDEN_STATE) for
                                            embedding in merged_embedding], dim=0)

            # Project to VL dimension -> (1, 49, H) (H is 512 for t5-small, 768 for t5-base)
            if self.lm != 'T5-Base':
                merged_embedding = self.img_projection_layer(merged_embedding)

            # Add modal type embedding to merged embedding
            merged_embedding += self.modal_embeddings(
                torch.ones((1, merged_embedding.shape[1]), dtype=torch.int, device=device))

            return merged_embedding

        def forward(self, text_enc, imgs, text_model):

            # Get the image embeddings (N x 1 x 49 x H)
            imgs_embedding = self.get_img_embedding(imgs)

            # Get the text embeddings (N x S x H)
            text_embeddings = text_model.get_input_embeddings()(text_enc)

            # Add modal embeddings to text
            text_embeddings += self.modal_embeddings(torch.zeros((1, text_embeddings.shape[1]), dtype=torch.int,
                                                                 device=device))

            # Concatenate embeddings -> (1 x S x 512)
            merged_embedding = torch.cat([text_embeddings, imgs_embedding], dim=1)

            return merged_embedding

    def forward(self, text_enc, imgs, labels=None):

        # Get the merged embeddings
        merged_embedding = self.mvp(text_enc, imgs, self.model)

        # If training include the labels
        return self.model(inputs_embeds=merged_embedding, labels=labels)

    def generate(self, text_enc, imgs, lidar=None):

        merged_embedding = self.mvp(text_enc, imgs, self.model)

        attention_mask = torch.ones(merged_embedding.shape[:2], dtype=torch.long, device=device)
        decoder_input_ids = torch.ones((merged_embedding.shape[0], 1), dtype=torch.long, device=device)*self.model.config.decoder_start_token_id
        output_ids = self.model.generate(attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, inputs_embeds=merged_embedding, max_length=512, early_stopping=True)

        return output_ids
