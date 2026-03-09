import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Single-branch 2D backbone that mirrors your SingleBranchEEGModel conv stacks ---
class _EEGBackbone(nn.Module):
    def __init__(self, input_channels: int, target_length: int):
        """
        input_channels: channels fed to Conv2d (usually 1)
        target_length:  time-length (W) after optional pooling, e.g., 400
        """
        super().__init__()
        self.target_length = target_length

        self.conv1_t = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.conv2_t = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.conv3_t = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.2),
        )
        self.conv4_t = nn.Sequential(
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(0.2),
        )
        self.conv5_t = nn.Sequential(
            nn.Conv2d(256, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )

        # Gate along time axis (W). We'll pool/reshape to [B, H, W] with W == target_length.
        self.fusion_weights = nn.Sequential(
            nn.Linear(self.target_length, self.target_length // 2),
            nn.ReLU(),
            nn.Linear(self.target_length // 2, self.target_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Accepts any of:
        - [B, C, H, W]         (preferred; C must equal input_channels)
        - [B, C, W]            -> interpreted as [B, C, 1, W]
        - [B, H, W]            -> interpreted as [B, H(=channels), 1, W]
        - [B, W]               -> interpreted as [B, 1, 1, W]
        Returns:
        feat_vec: [B, target_length]  (after gating + mean over H)
        """
        # --- Normalize to [B, C, H, W] ---
        if x.dim() == 4:
            # [B, C, H, W] or sometimes mistakenly [B, 1, C, W] (with C actually the channel count)
            B, C, H, W = x.shape
            # If channels stuck in H (e.g., [B, 1, 33, 400] while conv expects C=33), fix by swapping
            conv_in = self.conv1_t[0].in_channels  # expected input_channels for first conv
            if C != conv_in and H == conv_in:
                # Move H to channel dim -> [B, H, C, W] then set C=1 as height
                x = x.permute(0, 2, 1, 3).contiguous()  # [B, H, 1, W] with H == conv_in
        elif x.dim() == 3:
            # Treat the 2nd dim as channels -> [B, C, W] -> [B, C, 1, W]
            x = x.unsqueeze(2)
        elif x.dim() == 2:
            # [B, W] -> [B, 1, 1, W]
            x = x.unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # At this point x is [B, C, H, W]
        B, C, H, W = x.shape
        expected_C = self.conv1_t[0].in_channels
        if C != expected_C:
            raise RuntimeError(
                f"Backbone expected in_channels={expected_C} but got C={C} "
                f"(input shape {tuple(x.shape)}). If your channels are in the H dimension, "
                f"pass [B, C, W] or [B, C, 1, W]; this forward() already fixes the common "
                f"[B, 1, C, W] mistake by swapping dims."
            )

        # --- Time alignment: pool only along W to target_length, preserve H ---
        if W != self.target_length:
            x = F.adaptive_avg_pool2d(x, output_size=(H, self.target_length))

        # --- Convolutional stack ---
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        x = self.conv3_t(x)
        x = self.conv4_t(x)
        x = self.conv5_t(x)        # [B, 1, H, target_length]

        # --- Temporal gating and reduction ---
        x = x.squeeze(1)           # [B, H, target_length]
        gate = self.fusion_weights(x.mean(dim=1))  # [B, target_length]
        x = x * gate.unsqueeze(1)  # broadcast over H: [B, H, target_length]
        feat_vec = x.mean(dim=1)   # [B, target_length]
        return feat_vec



# --- Parallel model that uses the above SingleBranch-like backbone for each input ---
class LEEDNetModel(nn.Module):
    def __init__(self, input_channels_list, output_size, target_length=400):
        """
        input_channels_list: list of input channel counts per-branch (usually all 1)
        output_size:        number of classes
        target_length:      expected time-length for gating/classifier (e.g., 400)
        """
        super().__init__()
        self.output_size = output_size
        self.target_length = target_length

        # One backbone per branch (all mirror SingleBranchEEGModel conv stacks)
        self.branches = nn.ModuleList([
            _EEGBackbone(input_channels=ch, target_length=target_length)
            for ch in input_channels_list
        ])

        # Per-branch auxiliary heads (like your hybrid’s aux outputs)
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(target_length, 128),
                nn.GELU(),
                nn.GroupNorm(1, 128),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.GroupNorm(1, 64),
                nn.Dropout(0.4),
                nn.Linear(64, output_size)
            ) for _ in input_channels_list
        ])

        # Fused classifier over concatenated features (sum of target_length per branch)
        fused_dim = target_length * len(input_channels_list)

        # Optional global fusion gate over concatenated vector (mirrors your linear gate idea)
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Linear(fused_dim // 2, fused_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.GroupNorm(1, 128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.GroupNorm(1, 64),
            nn.Dropout(0.4),
            nn.Linear(64, output_size)
        )

    def forward(self, inputs, active_branch_indices=None):
        """
        inputs: list of tensors, one per branch.
                Each can be [B, 1, H, W], [B, 1, W], or [B, H, W]; W is time.
        active_branch_indices: which branches to use (default: all)
        """
        if active_branch_indices is None:
            active_branch_indices = list(range(len(self.branches)))

        feats = []
        aux_outputs = []

        for idx, (backbone, x) in enumerate(zip(self.branches, inputs)):
            if idx in active_branch_indices:
                feat_vec = backbone(x)                  # [B, target_length]
                aux = self.aux_heads[idx](feat_vec)     # [B, classes]
            else:
                B = x.size(0)
                feat_vec = torch.zeros(B, self.target_length, device=x.device)
                aux = torch.zeros(B, self.output_size, device=x.device)

            feats.append(feat_vec)
            aux_outputs.append(aux)

        # Concatenate per-branch features and apply a global gate (optional)
        x = torch.cat(feats, dim=1)                      # [B, fused_dim]
        gate = self.fusion_gate(x)                       # [B, fused_dim]
        x = gate * x

        out = self.classifier(x)                         # [B, classes]
        return out, aux_outputs
