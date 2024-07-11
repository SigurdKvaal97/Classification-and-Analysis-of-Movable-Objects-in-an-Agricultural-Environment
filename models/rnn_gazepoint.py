
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models import backbones

class FrameLSTM(nn.Module):

    def __init__(self, num_classes, max_len, hidden_size, pool_fn='L2', ant_loss=None):
        super().__init__()
        self.num_classes = num_classes
        self.max_len = max_len
        self.hidden_size = hidden_size

        self.pool_fn = pool_fn
        self.ant_loss = ant_loss

    def init_backbone(self, backbone):
        self.backbone = backbone()
        self.spatial_dim = self.backbone.spatial_dim
        self.rnn = nn.LSTM(self.backbone.feat_dim + 2, self.hidden_size, batch_first=True) # (B, T, num_maps)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

        feat_dim = self.backbone.feat_dim +2
        
        self.project = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(True),
                nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(True),
                )
        self.backbone_fn = backbone.__name__

        # LSTM hidden state
        n_layers = 1
        h0 = torch.zeros(n_layers, 1, self.hidden_size)
        c0 = torch.zeros(n_layers, 1, self.hidden_size)
        nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)

        print ('FrameLSTM created with out_ch: %d and spatial dim: %d'%(self.hidden_size, self.spatial_dim))

    def get_hidden_state(self, B, device):
        h = self.h0.expand(self.h0.shape[0], B, self.h0.shape[2]).contiguous().to(device)
        c = self.c0.expand(self.c0.shape[0], B, self.c0.shape[2]).contiguous().to(device)
        return (h, c)

    # (B, T, X) --> (B*T, X) --module--> (B*T, Y) --> (B, T, Y)
    def flatten_apply(self, tensor, module):
        shape = tensor.shape
        flat = (shape[0]*shape[1], ) + shape[2:]
        tensor = tensor.view(flat)
        out = module(tensor)
        uflat = (shape[0], shape[1], ) + out.shape[1:]
        out = out.view(uflat)
        return out

    
    def embed_clip(self, frame_feats, **kwargs):
        # sort clip features by length
        B = frame_feats.shape[0]
        
        # Ensure lengths is a t1D tensor of type int64 on CPU
        lengths = torch.tensor(kwargs['length'], dtype=torch.int64).cpu()
        
        # Use the modified lengths tensor
        packed_input = pack_padded_sequence(frame_feats, lengths, batch_first=True, enforce_sorted=False)
        
        
        self.rnn.flatten_parameters()  # This makes the internal data contiguous which might improve performance.
        packed_output, (hn, cn) = self.rnn(packed_input, self.get_hidden_state(B, frame_feats.device))
        
        clip_feats = hn[-1]  # Assuming you're only interested in the last hidden state of each sequence.
        
        # Decode the packed sequence. This is not necessarily required if you're only interested in clip_feats.
        output, _ = pad_packed_sequence(packed_output)
        output = output.transpose(0, 1)  # (B, T, hidden_dim) assuming rnn's output_size is hidden_dim
    
        return clip_feats, output

    def pool(self, frame_feats):
        if self.pool_fn=='L2':
            pool_feats = F.lp_pool2d(frame_feats, 2, self.spatial_dim)
        elif self.pool_fn=='avg':
            pool_feats = F.avg_pool2d(frame_feats, self.spatial_dim)
        return pool_feats

    def anticipation_loss(self, frame_feats, lstm_feats, batch):
        B = frame_feats.shape[0]
        T = lstm_feats.shape[1]

        positive, negative = batch['positive'], batch['negative']

        pos_gaze, neg_gaze = batch['pos_gaze'], batch['neg_gaze']
        
        target, length = batch['verb'], batch['length']
        

        # select the active frame from the clip
        lstm_preds = self.fc(lstm_feats) # (B, max_L, #classes)
        lstm_preds = lstm_preds.view(B*T, -1)
        target_flat = target.unsqueeze(1).expand(target.shape[0], T).contiguous().view(-1)
        pred_scores = -F.cross_entropy(lstm_preds, target_flat, reduction='none').view(B, T)
        
        _, frame_idx = pred_scores.max(1)
        frame_idx = torch.min(frame_idx, length-1) # don't select a padding frame!(0
 
        active_feats = frame_feats[torch.arange(B), frame_idx] # (B, 256, 28, 28)

        active_pooled = self.pool(active_feats).view(B, -1)

        
        def embed(x, pos_gaze):
            x = self.backbone(x)
           
            # Project the combined features
            pos_gaze_expanded = pos_gaze.unsqueeze(2).unsqueeze(3).expand(-1, -1, 28, 28)
            x_with_gaze = torch.cat((x, pos_gaze_expanded), dim=1)
            
            pred_frame = self.project(x_with_gaze)
            

            # Pool the projected features
            pooled = self.pool(pred_frame).view(B, -1)
            return pooled

        positive_pooled = embed(positive, pos_gaze)
        

        _, (hn, cn) = self.rnn(positive_pooled.unsqueeze(1), self.get_hidden_state(B, positive_pooled.device))
        preds = self.fc(hn[-1])
        
        aux_loss = F.cross_entropy(preds, target, reduction='none')

        if self.ant_loss=='mse':
            ant_loss = 0.1*((positive_pooled-active_pooled)**2).mean(1)
        elif self.ant_loss=='triplet':
            negative_pooled = self.backbone(negative)
            neg_gaze_expanded = neg_gaze.unsqueeze(2).unsqueeze(3).expand(-1, -1, 28, 28)
            negative_pooled= torch.cat((negative_pooled, neg_gaze_expanded), dim=1)
            negative_pooled = self.pool(negative_pooled).view(B, -1)

            anc, pos, neg = F.normalize(positive_pooled, 2), F.normalize(active_pooled, 2), F.normalize(negative_pooled, 2)
            
            ant_loss =  F.triplet_margin_loss(anc, pos, neg, margin=0.5, reduction='none')

        return {'ant_loss': ant_loss, 'aux_loss': aux_loss}



    def forward(self, batch, **kwargs):
        frames, length, gaze_data = batch['frames'], batch['length'], batch['gaze']  # Add gaze data here
        B, T = frames.shape[:2]
        S = self.spatial_dim
        
        
        frame_feats = self.flatten_apply(frames, lambda t: self.backbone(t))  # (B, T, 256, 28, 28)
        
        
        gaze_feat = gaze_data.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 28, 28)
        frame_feats = torch.cat((frame_feats, gaze_feat), dim=2)
        pool_feats = self.flatten_apply(frame_feats, lambda t: self.pool(t)).view(B, T, -1)  # (B, T, 256)
        
        clip_feats, lstm_feats = self.embed_clip(pool_feats, length=length)  # (B, 2048)
        
        preds = self.fc(clip_feats)
        
        loss_dict = {}
        target = batch['verb']
        cls_loss = F.cross_entropy(preds, target, reduction='none')

        loss_dict.update({'cls_loss': cls_loss})

        if not self.training or self.ant_loss is None:
            return preds, loss_dict

        loss_dict.update(self.anticipation_loss(frame_feats, lstm_feats, batch))

        return preds, loss_dict

    # forward/backward passes for class activation mapping
    def ic_features(self, image, **kwargs):
        feat = self.backbone(image)
        return feat.unsqueeze(1)
       

    def ic_classifier(self, frame_feats, **kwargs):
        
        B = frame_feats.shape[0]

        frame_feats = frame_feats.squeeze(1)
        
        
        gaze_point_example = [900, 300]
        max_x = 1920
        max_y = 1080
        normalized_gaze_point = [gaze_point_example[0] / max_x, gaze_point_example[1] / max_y]
        
        # Convert gaze_point_example to a tensor and expand it
        gaze_point_tensor = torch.tensor(normalized_gaze_point).view(1, 2, 1, 1).expand(frame_feats.size(0), 2, 28, 28)
        gaze_point_tensor = gaze_point_tensor.to(frame_feats.device)

        # Concatenate frame_feats and gaze_point_tensor along the second dimension
        frame_feats = torch.cat((frame_feats, gaze_point_tensor), dim=2)
        

        if self.ant_loss is not None:
            frame_feats = self.project(frame_feats)

        pool_feats = self.pool(frame_feats).view(B, -1)

        _, (hn, cn) = self.rnn(pool_feats.unsqueeze(1), self.get_hidden_state(B, pool_feats.device))
        pred = self.fc(hn[-1])
        return pred


def frame_lstm(num_classes, max_len, backbone, hidden_size=2050, ant_loss='mse'):
    net = FrameLSTM(num_classes, max_len, hidden_size, ant_loss=ant_loss)
    net.init_backbone(backbone)
    print ('Using backbone class: %s'%backbone)
    print ('Using ant loss fn: %s'%ant_loss)
    return net
    
