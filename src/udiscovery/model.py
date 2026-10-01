"""UrbanDE-Net model components.

The Graph WaveNet blocks are adapted from Zonghan Wu et al.'s MIT-licensed
Graph WaveNet implementation. See ``THIRD_PARTY_NOTICES.md`` and
``THIRD_PARTY_LICENSES/Graph-WaveNet-LICENSE`` in the repository root.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class NConv(nn.Module):
    """Neighborhood convolution used inside Graph WaveNet blocks."""

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, num_nodes, seq_len)
        # adjacency: (num_nodes, num_nodes)
        return torch.einsum('bcvt,vw->bcwt', (x, adjacency)).contiguous()


class GraphWaveNetGCN(nn.Module):
    """Spatial graph convolution from Graph WaveNet."""

    def __init__(self, c_in: int, c_out: int, dropout: float, support_len: int = 1, order: int = 2):
        super().__init__()
        self.nconv = NConv()
        self.order = order
        input_channels = (order * support_len + 1) * c_in
        self.mlp = nn.Conv2d(input_channels, c_out, kernel_size=(1, 1))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        if supports is None or len(supports) == 0:
            raise ValueError("supports must contain at least one adjacency matrix")

        out = [x]
        for support in supports:
            x1 = self.nconv(x, support)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, support)
                out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GraphWaveNetEncoder(nn.Module):
    """Graph WaveNet encoder that produces node- and graph-level embeddings."""

    def __init__(
        self,
        num_nodes: int,
        sequence_length: int,
        output_dim: int,
        input_channels: int = 1,
        adjacency_mask: Optional[torch.Tensor] = None,
        distance_matrix: Optional[torch.Tensor] = None,
        dropout: float = 0.1,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        init_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.dropout = dropout
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj and gcn_bool
        self.init_seed = init_seed

        # Build supports from adjacency and distance information and register buffers
        (
            edge_mask,
            raw_edge_weights,
            normalized_support,
            identity_support,
        ) = self._build_supports(adjacency_mask, distance_matrix)

        if edge_mask is not None:
            self.register_buffer('edge_mask', edge_mask)
        else:
            self.edge_mask = None

        if raw_edge_weights is not None:
            self.register_buffer('edge_weight_values', raw_edge_weights)
        else:
            self.register_buffer('edge_weight_values', torch.empty(0))

        if normalized_support is not None:
            self.register_buffer('static_support', normalized_support)
            self.has_static_support = True
        else:
            self.register_buffer('static_support', torch.empty(0))
            self.has_static_support = False

        self.register_buffer('identity_support', identity_support)

        base_supports = 1  # identity always included
        if self.has_static_support:
            base_supports += 1
        self.supports_len = base_supports + (1 if self.addaptadj else 0)

        # Graph WaveNet components
        in_channels = self.input_channels
        self.start_conv = nn.Conv2d(in_channels, residual_channels, kernel_size=(1, 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1))
                )
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))

                if self.gcn_bool:
                    self.gconv.append(
                        GraphWaveNetGCN(
                            dilation_channels,
                            residual_channels,
                            dropout,
                            support_len=self.supports_len,
                        )
                    )
                else:
                    self.gconv.append(None)

                dilation *= 2
                self.receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.dropout_layer = nn.Dropout(dropout)
        self.output_proj = nn.Linear(end_channels, output_dim)
        self.global_attn = nn.Linear(output_dim, 1)

        if self.addaptadj:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
        else:
            self.register_parameter('nodevec1', None)
            self.register_parameter('nodevec2', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_seed is not None:
            current_state = torch.get_rng_state()
            torch.manual_seed(self.init_seed)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.addaptadj:
            nn.init.xavier_normal_(self.nodevec1)
            nn.init.xavier_normal_(self.nodevec2)

        if self.init_seed is not None:
            torch.set_rng_state(current_state)

    def _build_supports(
        self,
        adjacency_mask: Optional[torch.Tensor],
        distance_matrix: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        if adjacency_mask is None:
            identity = torch.eye(self.num_nodes, dtype=torch.float32)
            return None, None, None, identity

        adjacency = adjacency_mask.float().clone()
        adjacency.fill_diagonal_(0.0)

        if distance_matrix is not None:
            distance = distance_matrix.float().clone()
            eps = 1e-6
            weights = torch.where(
                adjacency > 0,
                1.0 / (distance + eps),
                torch.zeros_like(distance),
            )
        else:
            weights = adjacency

        weights = (weights + weights.t()) / 2.0
        weights.fill_diagonal_(0.0)

        degree = torch.sum(weights, dim=1)
        deg_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        normalized = deg_inv_sqrt.unsqueeze(1) * weights * deg_inv_sqrt.unsqueeze(0)

        edge_mask = adjacency
        edge_values = weights[edge_mask > 0].flatten()
        identity = torch.eye(self.num_nodes, dtype=torch.float32)

        if edge_values.numel() == 0:
            edge_values = torch.empty(0)

        return edge_mask, edge_values, normalized, identity

    def forward(
        self,
        x: torch.Tensor,
        return_node_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)

        if x.dim() == 2:
            x = x.view(batch_size, self.num_nodes, self.sequence_length, self.input_channels)
        elif x.dim() == 4:
            if x.size(1) != self.input_channels:
                raise ValueError(
                    f"Expected input with {self.input_channels} channels, got {x.size(1)}"
                )
            x = x.permute(0, 2, 3, 1)  # (batch, num_nodes, seq_len, channels)
        else:
            raise ValueError("GraphWaveNetEncoder expects input of shape (batch, features) or (batch, channels, num_nodes, seq_len)")

        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, channels, num_nodes, seq_len)

        seq_len = x.size(3)
        if seq_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - seq_len, 0, 0, 0))

        x = self.start_conv(x)
        skip = None

        supports: Optional[List[torch.Tensor]] = None
        if self.gcn_bool:
            supports = []
            if self.has_static_support:
                supports.append(self.static_support)
            supports.append(self.identity_support)
            if self.addaptadj:
                adaptive_adj = F.softmax(
                    F.relu(torch.matmul(self.nodevec1, self.nodevec2)), dim=1
                )
                supports.append(adaptive_adj)

        total_layers = len(self.filter_convs)
        for layer_idx in range(total_layers):
            residual = x
            filter_out = torch.tanh(self.filter_convs[layer_idx](residual))
            gate_out = torch.sigmoid(self.gate_convs[layer_idx](residual))
            x = filter_out * gate_out

            s = self.skip_convs[layer_idx](x)
            if skip is None:
                skip = s
            else:
                skip = skip[..., -s.size(3):] + s

            if self.gcn_bool and supports is not None:
                x = self.gconv[layer_idx](x, supports)
            else:
                x = self.residual_convs[layer_idx](x)

            x = x + residual[..., -x.size(3):]
            x = self.bn[layer_idx](x)

        if skip is None:
            skip = x

        skip = F.relu(skip)
        skip = F.relu(self.end_conv_1(skip))
        skip = self.dropout_layer(skip)

        node_map = skip.mean(dim=3)  # (batch, channels, num_nodes)
        node_map = node_map.permute(0, 2, 1)  # (batch, num_nodes, channels)
        node_embeddings = self.output_proj(node_map)

        att_scores = self.global_attn(node_embeddings)
        att_weights = torch.softmax(att_scores, dim=1)
        global_embedding = torch.sum(att_weights * node_embeddings, dim=1)

        if return_node_embeddings:
            return node_embeddings, global_embedding
        return global_embedding

    def get_edge_stats(self) -> Dict[str, Union[float, int, str]]:
        edge_values = self.edge_weight_values
        if edge_values.numel() == 0:
            return {'num_edges': 0, 'edge_weight_min': 'None', 'edge_weight_max': 'None', 'edge_weight_mean': 'None'}
        return {
            'num_edges': edge_values.numel(),
            'edge_weight_min': edge_values.min().item(),
            'edge_weight_max': edge_values.max().item(),
            'edge_weight_mean': edge_values.mean().item(),
        }


class MLP(nn.Module):
    """Simple MLP with configurable layers and activation"""
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout=0.1, init_seed=None):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.LayerNorm(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.LayerNorm(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(self._get_activation(activation))

        self.network = nn.Sequential(*layers)

        # Store init_seed and apply weight initialization
        self.init_seed = init_seed
        self.apply(self._init_weights)

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'softplus':
            return nn.Softplus()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'linear':
            return nn.Identity()
        else:
            return nn.ReLU()

    def _init_weights(self, module):
        """Initialize weights with optional seed control"""
        if isinstance(module, nn.Linear):
            # Set global seed state if seed is provided (compatible with older PyTorch versions)
            if self.init_seed is not None:
                # Save current random state
                current_state = torch.get_rng_state()
                # Set seed for this initialization
                torch.manual_seed(self.init_seed)
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                # Restore previous random state
                torch.set_rng_state(current_state)
            else:
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


class NodewiseMLP(nn.Module):
    """MLP that processes node-wise features with node+global embedding fusion and per-node bias"""
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout=0.1, init_seed=None, n_nodes=None):
        super().__init__()

        # Store parameters
        self.input_size = input_size  # Expected to be concat_dim = node_dim + global_dim
        self.output_size = output_size  # Parameters per node
        self.n_nodes = n_nodes

        # Build network body (all layers except final projection)
        body_layers = []

        # Input layer
        body_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        body_layers.append(nn.LayerNorm(hidden_sizes[0]))
        body_layers.append(nn.ReLU())
        body_layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            body_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            body_layers.append(nn.LayerNorm(hidden_sizes[i+1]))
            body_layers.append(nn.ReLU())
            body_layers.append(nn.Dropout(dropout))

        self.network_body = nn.Sequential(*body_layers)

        # Final linear projection (shared across nodes)
        self.final_linear = nn.Linear(hidden_sizes[-1], output_size)

        # Output activation
        self.activation = self._get_activation(activation)

        # Optional per-node bias (pre-activation), shape: (n_nodes, output_size)
        if n_nodes is not None:
            self.per_node_bias = nn.Parameter(torch.zeros(n_nodes, output_size))
        else:
            self.per_node_bias = None

        # Store init_seed and apply weight initialization
        self.init_seed = init_seed
        self.apply(self._init_weights)

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'softplus':
            return nn.Softplus()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'linear':
            return nn.Identity()
        else:
            return nn.ReLU()

    def _init_weights(self, module):
        """Initialize weights with optional seed control"""
        if isinstance(module, nn.Linear):
            # Set global seed state if seed is provided (compatible with older PyTorch versions)
            if self.init_seed is not None:
                current_state = torch.get_rng_state()
                torch.manual_seed(self.init_seed)
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                torch.set_rng_state(current_state)
            else:
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass for node-wise parameter generation
        Args:
            x: Input tensor of shape (batch_size, n_nodes, input_size) where input_size = node_dim + global_dim
        Returns:
            output: Tensor of shape (batch_size, n_nodes, output_size) - parameters for each node
        """
        batch_size, n_nodes, input_dim = x.shape

        # Reshape to (batch_size * n_nodes, input_dim) for parallel processing
        x_reshaped = x.view(batch_size * n_nodes, input_dim)

        # Body forward
        h = self.network_body(x_reshaped)  # (batch_size * n_nodes, hidden)

        # Final projection
        z = self.final_linear(h)  # (batch_size * n_nodes, output_size)
        z = z.view(batch_size, n_nodes, self.output_size)  # reshape to per-node

        # Add per-node bias pre-activation if available
        if self.per_node_bias is not None:
            z = z + self.per_node_bias.unsqueeze(0)

        # Apply activation
        output = self.activation(z)
        return output


@dataclass
class UrbanDENetOutput:
    """Parameter fields inferred from a batch of state-history windows."""

    interaction_parameters: Dict[str, torch.Tensor]
    temporal_sin: torch.Tensor
    temporal_cos: torch.Tensor


class UrbanDENet(nn.Module):
    """Portable UrbanDE-Net parameter estimator used by the release trainer.

    A Graph WaveNet encoder maps each ``(history, node)`` observation window to
    node- and graph-level embeddings. Node-scoped interaction parameters and
    Fourier coefficients use nodewise heads; parameters with global scope use
    one shared graph-level head. The governing equation itself remains explicit
    in :mod:`udiscovery.equations`.
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        history_length: int,
        parameter_names: Sequence[str],
        parameter_scopes: Sequence[str],
        adjacency_mask: torch.Tensor,
        distance_matrix: torch.Tensor,
        temporal_harmonics: int = 1,
        encoder_dim: int = 128,
        head_hidden: Sequence[int] = (128, 64),
        encoder_dropout: float = 0.1,
        head_dropout: float = 0.1,
        residual_channels: int = 32,
        dilation_channels: Optional[int] = None,
        skip_channels: int = 64,
        end_channels: int = 128,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        adaptive_adjacency: bool = True,
        initial_parameters: Optional[Mapping[str, object]] = None,
        bounded_parameter_names: Sequence[str] = (),
        init_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2")
        if history_length < 2:
            raise ValueError("history_length must be at least 2")
        if temporal_harmonics < 0:
            raise ValueError("temporal_harmonics must be non-negative")
        if len(parameter_names) != len(parameter_scopes):
            raise ValueError("parameter_names and parameter_scopes must have equal length")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter_names must be unique")
        if any(scope not in {"node", "global"} for scope in parameter_scopes):
            raise ValueError("parameter scopes must be 'node' or 'global'")
        if not head_hidden:
            raise ValueError("head_hidden must contain at least one width")

        self.num_nodes = int(num_nodes)
        self.history_length = int(history_length)
        self.temporal_harmonics = int(temporal_harmonics)
        self.parameter_names = tuple(parameter_names)
        self.parameter_scopes = tuple(parameter_scopes)
        self.node_parameter_names = tuple(
            name for name, scope in zip(self.parameter_names, self.parameter_scopes)
            if scope == "node"
        )
        self.global_parameter_names = tuple(
            name for name, scope in zip(self.parameter_names, self.parameter_scopes)
            if scope == "global"
        )
        self.bounded_parameter_names = frozenset(bounded_parameter_names)
        unknown_bounded = self.bounded_parameter_names.difference(self.parameter_names)
        if unknown_bounded:
            raise ValueError(f"bounded parameters are not model outputs: {sorted(unknown_bounded)}")

        dilation_channels = residual_channels if dilation_channels is None else dilation_channels
        self.encoder = GraphWaveNetEncoder(
            num_nodes=self.num_nodes,
            sequence_length=self.history_length,
            output_dim=int(encoder_dim),
            input_channels=1,
            adjacency_mask=adjacency_mask,
            distance_matrix=distance_matrix,
            dropout=float(encoder_dropout),
            residual_channels=int(residual_channels),
            dilation_channels=int(dilation_channels),
            skip_channels=int(skip_channels),
            end_channels=int(end_channels),
            kernel_size=int(kernel_size),
            blocks=int(blocks),
            layers=int(layers),
            addaptadj=bool(adaptive_adjacency),
            init_seed=init_seed,
        )

        fused_dim = 2 * int(encoder_dim)
        seed_base = 0 if init_seed is None else int(init_seed)
        self.interaction_node_head: Optional[NodewiseMLP]
        if self.node_parameter_names:
            self.interaction_node_head = NodewiseMLP(
                input_size=fused_dim,
                hidden_sizes=list(head_hidden),
                output_size=len(self.node_parameter_names),
                activation="linear",
                dropout=float(head_dropout),
                init_seed=None if init_seed is None else seed_base + 1,
                n_nodes=self.num_nodes,
            )
        else:
            self.interaction_node_head = None

        self.interaction_global_head: Optional[MLP]
        if self.global_parameter_names:
            self.interaction_global_head = MLP(
                input_size=int(encoder_dim),
                hidden_sizes=list(head_hidden),
                output_size=len(self.global_parameter_names),
                activation="linear",
                dropout=float(head_dropout),
                init_seed=None if init_seed is None else seed_base + 2,
            )
        else:
            self.interaction_global_head = None

        self.temporal_head = NodewiseMLP(
            input_size=fused_dim,
            hidden_sizes=list(head_hidden),
            output_size=2 * self.temporal_harmonics,
            activation="linear",
            dropout=float(head_dropout),
            init_seed=None if init_seed is None else seed_base + 3,
            n_nodes=self.num_nodes,
        ) if self.temporal_harmonics else None
        self._initialize_output_heads(initial_parameters or {})

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(min=1.0e-6)
        return value + torch.log(-torch.expm1(-value))

    @staticmethod
    def _logit(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(min=1.0e-6, max=1.0 - 1.0e-6)
        return torch.log(value) - torch.log1p(-value)

    def _raw_initial(self, name: str, value: object, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"initial parameter {name!r} contains non-finite values")
        if name in self.bounded_parameter_names:
            return self._logit(tensor)
        if torch.any(tensor <= 0):
            raise ValueError(f"initial parameter {name!r} must be positive")
        return self._inverse_softplus(tensor)

    @staticmethod
    def _last_linear(module: nn.Module) -> nn.Linear:
        linear_layers = [child for child in module.modules() if isinstance(child, nn.Linear)]
        if not linear_layers:
            raise RuntimeError("parameter head has no linear output layer")
        return linear_layers[-1]

    def _initialize_output_heads(self, initial_parameters: Mapping[str, object]) -> None:
        """Start heads at configured constants while retaining learnable weights."""

        with torch.no_grad():
            if self.interaction_node_head is not None:
                layer = self.interaction_node_head.final_linear
                layer.weight.zero_()
                layer.bias.zero_()
                if self.interaction_node_head.per_node_bias is not None:
                    self.interaction_node_head.per_node_bias.zero_()
                for index, name in enumerate(self.node_parameter_names):
                    raw = self._raw_initial(
                        name,
                        initial_parameters.get(name, 1.0),
                        device=layer.bias.device,
                        dtype=layer.bias.dtype,
                    )
                    if raw.ndim == 0 or raw.numel() == 1:
                        layer.bias[index] = raw.reshape(())
                    elif tuple(raw.shape) == (self.num_nodes,):
                        self.interaction_node_head.per_node_bias[:, index].copy_(raw)
                    else:
                        raise ValueError(
                            f"initial node parameter {name!r} must be scalar or shape "
                            f"({self.num_nodes},), got {tuple(raw.shape)}"
                        )

            if self.interaction_global_head is not None:
                layer = self._last_linear(self.interaction_global_head)
                layer.weight.zero_()
                layer.bias.zero_()
                for index, name in enumerate(self.global_parameter_names):
                    raw = self._raw_initial(
                        name,
                        initial_parameters.get(name, 1.0),
                        device=layer.bias.device,
                        dtype=layer.bias.dtype,
                    )
                    if raw.numel() != 1:
                        raise ValueError(f"initial global parameter {name!r} must be scalar")
                    layer.bias[index] = raw.reshape(())

            if self.temporal_head is not None:
                temporal_layer = self.temporal_head.final_linear
                temporal_layer.weight.zero_()
                temporal_layer.bias.zero_()
                if self.temporal_head.per_node_bias is not None:
                    self.temporal_head.per_node_bias.zero_()

    def _transform(self, names: Sequence[str], raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        transformed: Dict[str, torch.Tensor] = {}
        for index, name in enumerate(names):
            value = raw[..., index]
            if name in self.bounded_parameter_names:
                value = torch.sigmoid(value)
            else:
                value = F.softplus(value) + 1.0e-6
            transformed[name] = value
        return transformed

    def forward(self, histories: torch.Tensor) -> UrbanDENetOutput:
        """Infer equation and temporal parameters from ``(B, history, N)`` windows."""

        if histories.ndim != 3:
            raise ValueError(
                "UrbanDENet expects histories with shape (batch, history_length, num_nodes)"
            )
        expected = (self.history_length, self.num_nodes)
        if tuple(histories.shape[1:]) != expected:
            raise ValueError(
                f"Expected history trailing shape {expected}, got {tuple(histories.shape[1:])}"
            )
        if not histories.dtype.is_floating_point:
            raise ValueError("histories must use a floating-point dtype")

        encoder_input = histories.transpose(1, 2).unsqueeze(1)
        node_embeddings, global_embedding = self.encoder(
            encoder_input, return_node_embeddings=True
        )
        expanded_global = global_embedding.unsqueeze(1).expand(
            histories.size(0), self.num_nodes, global_embedding.size(-1)
        )
        fused = torch.cat([node_embeddings, expanded_global], dim=-1)

        interaction: Dict[str, torch.Tensor] = {}
        if self.interaction_node_head is not None:
            interaction.update(
                self._transform(self.node_parameter_names, self.interaction_node_head(fused))
            )
        if self.interaction_global_head is not None:
            interaction.update(
                self._transform(
                    self.global_parameter_names,
                    self.interaction_global_head(global_embedding),
                )
            )

        if self.temporal_head is None:
            temporal_sin = histories.new_zeros((histories.shape[0], self.num_nodes, 0))
            temporal_cos = histories.new_zeros((histories.shape[0], self.num_nodes, 0))
        else:
            temporal = self.temporal_head(fused)
            temporal_sin = temporal[..., 0::2]
            temporal_cos = temporal[..., 1::2]
        return UrbanDENetOutput(interaction, temporal_sin, temporal_cos)

    def architecture_metadata(self) -> Dict[str, object]:
        """Return JSON-safe metadata needed to rebuild this model for inference."""

        return {
            "class": type(self).__name__,
            "num_nodes": self.num_nodes,
            "history_length": self.history_length,
            "temporal_harmonics": self.temporal_harmonics,
            "parameter_names": list(self.parameter_names),
            "parameter_scopes": list(self.parameter_scopes),
            "node_parameter_names": list(self.node_parameter_names),
            "global_parameter_names": list(self.global_parameter_names),
            "bounded_parameter_names": sorted(self.bounded_parameter_names),
            "encoder_output_dim": self.encoder.output_dim,
            "encoder_receptive_field": self.encoder.receptive_field,
        }
