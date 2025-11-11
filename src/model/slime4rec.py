import math
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward



class SLIME4RecModel(SequentialRecModel):
    def __init__(self, args):
        super(SLIME4RecModel, self).__init__(args)

        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = SLIME4RecEncoder(args)
        self.batch_size = args.batch_size
        self.gamma = 1e-10

        # arguments for SLIME4rec
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.tau = args.tau

        self.ssl = args.ssl
        self.sim = args.sim
        self.lmd_sem = args.lmd_sem
        self.lmd = args.lmd

        self.apply(self.init_weights)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask


    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):

        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        z = z[:, -1, :]

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels



    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]

        # cross-entropy loss
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        # Unsupervised NCE: original vs dropout

        if self.ssl in ['us', 'un']:
            aug_seq_output = self.forward(input_ids)
            nce_logits, nce_labels = self.info_nce(seq_output, aug_seq_output, temp=self.tau,
                                                   batch_size=input_ids.shape[0], sim=self.sim)
            loss += self.lmd * self.aug_nce_fct(nce_logits, nce_labels)

        # Supervised NCE: original vs semantic augmentation
        if self.ssl in ['us', 'su']:
            sem_aug = same_target
            sem_aug_seq_output = self.forward(sem_aug)
            sem_nce_logits, sem_nce_labels = self.info_nce(seq_output, sem_aug_seq_output, temp=self.tau,
                                                           batch_size=input_ids.shape[0], sim=self.sim)
            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

        # Unsupervised + Supervised NCE: dropout vs semantic augmentation
        if self.ssl == 'us_x':
            # unsupervised
            aug_seq_output = self.forward(input_ids)
            # supervised
            sem_aug = same_target
            sem_aug_seq_output = self.forward(sem_aug)

            sem_nce_logits, sem_nce_labels = self.info_nce(aug_seq_output, sem_aug_seq_output, temp=self.tau,
                                                           batch_size=input_ids.shape[0], sim=self.sim)

            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

        return loss


class SLIME4RecEncoder(nn.Module):
    def __init__(self, args):
        super(SLIME4RecEncoder, self).__init__()
        self.args = args

        self.blocks = []
        for i in range(args.num_hidden_layers):
            self.blocks.append(SLIME4RecBlock(args, layer_num=i))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):

        all_encoder_layers = [hidden_states]

        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)  # hidden_states => torch.Size([256, 50, 64])

        return all_encoder_layers


class SLIME4RecBlock(nn.Module):
    def __init__(self, args, layer_num):
        super(SLIME4RecBlock, self).__init__()
        self.layer = SLIME4RecLayer(args, layer_num)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class SLIME4RecLayer(nn.Module):
    def __init__(self, args, i):
        super(SLIME4RecLayer, self).__init__()

        self.dropout = nn.Dropout(0.1)
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.max_item_list_length = args.max_seq_length
        self.n_layers = args.num_hidden_layers

        self.scale = None
        self.mask_flag = True
        self.output_attention = False
        self.filter_mixer = args.filter_mixer
        self.residual = True
        self.complex_weight = nn.Parameter(
            torch.randn(1, self.max_item_list_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        if self.filter_mixer == 'G':
            self.complex_weight_G = nn.Parameter(
                torch.randn(1, self.max_item_list_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        elif self.filter_mixer == 'L':
            self.complex_weight_L = nn.Parameter(
                torch.randn(1, self.max_item_list_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        elif self.filter_mixer == 'M':
            self.complex_weight_G = nn.Parameter(
                torch.randn(1, self.max_item_list_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
            self.complex_weight_L = nn.Parameter(
                torch.randn(1, self.max_item_list_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)


        
        self.dynamic_ratio = args.dynamic_ratio

        self.slide_step = ((self.max_item_list_length // 2 + 1) * (1 - self.dynamic_ratio)) // (self.n_layers - 1)

        self.static_ratio = 1 / self.n_layers
        self.filter_size = self.static_ratio * (self.max_item_list_length // 2 + 1)
        self.slide_mode =  args.slide_mode

        if self.slide_mode == 'one':
            G_i = i
            L_i = self.n_layers - 1 - i
        elif self.slide_mode == 'two':
            G_i = self.n_layers - 1 - i
            L_i = i
        elif self.slide_mode == 'three':
            G_i = self.n_layers - 1 - i
            L_i = self.n_layers - 1 - i
        elif self.slide_mode == 'four':
            G_i = i
            L_i = i

        # print("slide_mode:", self.slide_mode, len(self.slide_mode), type(self.slide_mode))


        if self.filter_mixer == 'G' or self.filter_mixer == 'M':
            self.w = self.dynamic_ratio
            self.s = self.slide_step
            if self.filter_mixer == 'M':
                self.G_left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (G_i * self.s))
                self.G_right = int((self.max_item_list_length // 2 + 1) - G_i * self.s)
            self.left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (G_i * self.s))
            self.right = int((self.max_item_list_length // 2 + 1) - G_i * self.s)


        if self.filter_mixer == 'L' or self.filter_mixer == 'M':
            self.w = self.static_ratio
            self.s = self.filter_size
            if self.filter_mixer == 'M':
                self.L_left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (L_i * self.s))
                self.L_right = int((self.max_item_list_length // 2 + 1) - L_i * self.s)

            self.left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (L_i * self.s))
            self.right = int((self.max_item_list_length // 2 + 1) - L_i * self.s)
            print("====================================================================================G_left, right",
                  self.G_left, self.G_right, self.G_right - self.G_left)
            print("====================================================================================L_left, Light",
                  self.L_left, self.L_right, self.L_right - self.L_left)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # [256, 50, 2, 32]
        return x

    def forward(self, input_tensor, attention_mask):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        if self.filter_mixer == 'M':
            weight_g = torch.view_as_complex(self.complex_weight_G)
            weight_l = torch.view_as_complex(self.complex_weight_L)
            G_x = x
            L_x = x.clone()
            G_x[:, :self.G_left, :] = 0
            G_x[:, self.G_right:, :] = 0
            output = G_x * weight_g

            L_x[:, :self.L_left, :] = 0
            L_x[:, self.L_right:, :] = 0
            output += L_x * weight_l


        else:
            weight = torch.view_as_complex(self.complex_weight)
            x[:, :self.left, :] = 0
            x[:, self.right:, :] = 0
            output = x * weight

        sequence_emb_fft = torch.fft.irfft(output, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)

        if self.residual:
            origianl_out = self.LayerNorm(hidden_states + input_tensor)
        else:
            origianl_out = self.LayerNorm(hidden_states)

        return origianl_out
