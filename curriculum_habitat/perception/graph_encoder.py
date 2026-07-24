"""Metric scene-graph encoder (GIROL port, HM3D).

Consumes the fixed transport contract

    graph_flat : [B, N*6]           (N = NUM_NODES = 28)
    per node   : [object_id, active, is_goal, x, y, z]

and produces a [B, 128] graph embedding for the Q-network + orientation aux head.

Coordinate convention (see MetricGraphBuilder): x, y are the FLOOR plane, z is height
(GIROL / Isaac convention). Directions/distances on the edges are computed in the (x, y)
plane; z is only ever a node feature.

This is the METRIC encoder from GRAPHS.md. The very same class also serves the flat
non-metric variant (#3) by flipping ``include_node_metric=False`` — the only difference
is whether the raw (x,y,z) node block is fed in.

    node features (each -> 32, concat -> MLP -> 128):
        name_embedding      CLIP(name) 512 -> 128 -> 32     (always)
        xyz_metric_embedding raw (x,y,z) 3 -> 32 -> 32      (include_node_metric only)
        is_goal_embedding    is_goal 1 -> 32                (off by default)
    edges (edge_dim = 10, fixed one-hot):
        direction (6): same, in_front_of, behind, left_of, right_of, self
        distance  (4): very_close<5m, close<10m, far<15m, very_far<=20m   (per axis)
        topology: goal-star — goal -> every active non-goal object, TWO parallel edges
                  (x-axis relation + y-axis relation) + self-loops.
    encoder: node_mlp(->128) -> 2x GATv2Conv(heads=2, edge_dim=10, LayerNorm)
    readout: concat[mean_pool, max_pool, attention_pool] (384) -> 128 -> 128
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

NUM_NODES = 31  # 1 goal + up to 30 context objects
NODE_FIELDS = 6  # object_id, active, is_goal, x, y, z
CLIP_DIM = 512
BLOCK_DIM = 32
HIDDEN = 128
EDGE_DIM = 10  # 6 direction + 4 distance
GAT_HEADS = 2
GAT_OUT = HIDDEN // GAT_HEADS  # concat(heads) -> HIDDEN

# Direction one-hot slots.
DIR_SAME, DIR_FRONT, DIR_BEHIND, DIR_LEFT, DIR_RIGHT, DIR_SELF = range(6)
# Distance bucket edges (metres); bucketize -> {0:<5, 1:<10, 2:<15, 3:>=15 (<=20)}.
DIST_BOUNDS = (5.0, 10.0, 15.0)
ALIGN_THRESHOLD = 0.5  # |delta| below this on an axis => "same" (aligned)


class MetricGraphEncoder(nn.Module):
    def __init__(
        self,
        clip_text_embeddings: Optional[torch.Tensor] = None,
        vocab_size: Optional[int] = None,
        include_node_metric: bool = True,
        include_is_goal_feature: bool = False,
        num_nodes: int = NUM_NODES,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.include_node_metric = include_node_metric
        self.include_is_goal_feature = include_is_goal_feature

        # object_id -> frozen CLIP name vector. Falls back to a learnable table when no
        # CLIP cache is supplied (keeps the module runnable in isolation / for tests).
        if clip_text_embeddings is not None:
            self.register_buffer("clip_table", clip_text_embeddings.float(), persistent=True)
            vocab_size = clip_text_embeddings.shape[0]
        else:
            assert vocab_size is not None, "pass clip_text_embeddings or vocab_size"
            self.clip_table = nn.Parameter(torch.randn(vocab_size, CLIP_DIM) * 0.02)
        self.vocab_size = vocab_size

        self.name_proj = nn.Sequential(
            nn.Linear(CLIP_DIM, 128), nn.ReLU(inplace=True), nn.Linear(128, BLOCK_DIM)
        )
        if include_node_metric:
            self.xyz_proj = nn.Sequential(
                nn.Linear(3, BLOCK_DIM), nn.ReLU(inplace=True), nn.Linear(BLOCK_DIM, BLOCK_DIM)
            )
        if include_is_goal_feature:
            self.goal_proj = nn.Linear(1, BLOCK_DIM)

        node_in = BLOCK_DIM * (1 + int(include_node_metric) + int(include_is_goal_feature))
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, HIDDEN), nn.ReLU(inplace=True), nn.Linear(HIDDEN, HIDDEN)
        )

        self.gat1 = GATv2Conv(HIDDEN, GAT_OUT, heads=GAT_HEADS, edge_dim=EDGE_DIM, add_self_loops=False)
        self.gat2 = GATv2Conv(HIDDEN, GAT_OUT, heads=GAT_HEADS, edge_dim=EDGE_DIM, add_self_loops=False)
        self.norm1 = nn.LayerNorm(HIDDEN)
        self.norm2 = nn.LayerNorm(HIDDEN)

        self.attn_score = nn.Linear(HIDDEN, 1)
        self.readout = nn.Sequential(
            nn.Linear(HIDDEN * 3, HIDDEN), nn.ReLU(inplace=True), nn.Linear(HIDDEN, HIDDEN)
        )

    # ---- node features -------------------------------------------------------
    def _node_features(self, object_id, active, is_goal, xyz):
        """[B, N, *] fields -> node embedding [B, N, HIDDEN]."""
        name_vec = self.clip_table[object_id.clamp(0, self.vocab_size - 1)]  # [B,N,512]
        blocks = [self.name_proj(name_vec)]
        if self.include_node_metric:
            blocks.append(self.xyz_proj(xyz))
        if self.include_is_goal_feature:
            blocks.append(self.goal_proj(is_goal.unsqueeze(-1)))
        node_in = torch.cat(blocks, dim=-1)
        node_in = node_in * active.unsqueeze(-1)  # zero out padding slots
        return self.node_mlp(node_in)

    # ---- edge features -------------------------------------------------------
    @staticmethod
    def _axis_edge_attr(delta, axis):
        """delta: [K] signed goal->object offset on one floor axis -> [K, EDGE_DIM]."""
        k = delta.shape[0]
        attr = delta.new_zeros((k, EDGE_DIM))
        absd = delta.abs()
        same = absd < ALIGN_THRESHOLD
        positive = delta > 0
        direction = torch.full((k,), DIR_SAME, dtype=torch.long, device=delta.device)
        if axis == "x":
            direction[~same & positive] = DIR_RIGHT
            direction[~same & ~positive] = DIR_LEFT
        else:  # "y"
            direction[~same & positive] = DIR_FRONT
            direction[~same & ~positive] = DIR_BEHIND
        attr[torch.arange(k), direction] = 1.0
        bucket = torch.bucketize(absd, absd.new_tensor(DIST_BOUNDS)).clamp(max=3)
        attr[torch.arange(k), 6 + bucket] = 1.0
        return attr

    def _build_batch(self, node_h, active, is_goal, xyz):
        """Assemble a PyG batch of goal-star graphs (active nodes only)."""
        xy = xyz[..., :2]  # floor plane
        data_list = []
        for b in range(node_h.shape[0]):
            amask = active[b].bool()
            nidx = amask.nonzero(as_tuple=False).squeeze(1)
            if nidx.numel() == 0:
                data_list.append(Data(
                    x=node_h[b, :1], edge_index=torch.zeros(2, 0, dtype=torch.long, device=node_h.device),
                    edge_attr=node_h.new_zeros((0, EDGE_DIM))))
                continue
            g2l = torch.full((self.num_nodes,), -1, dtype=torch.long, device=node_h.device)
            g2l[nidx] = torch.arange(nidx.numel(), device=node_h.device)
            x = node_h[b, nidx]
            xy_b = xy[b, nidx]

            n = nidx.numel()
            src = torch.arange(n, device=node_h.device)
            # self-loops: direction=self, distance=very_close.
            self_attr = node_h.new_zeros((n, EDGE_DIM))
            self_attr[:, DIR_SELF] = 1.0
            self_attr[:, 6 + 0] = 1.0
            edge_index = [torch.stack([src, src])]
            edge_attr = [self_attr]

            goal_locals = g2l[(is_goal[b].bool() & amask).nonzero(as_tuple=False).squeeze(1)]
            for gl in goal_locals.tolist():
                others = src[src != gl]
                if others.numel() == 0:
                    continue
                delta = xy_b[others] - xy_b[gl]  # [K,2]
                gl_row = torch.full_like(others, gl)
                # two parallel edges goal->object: x-axis relation and y-axis relation.
                edge_index.append(torch.stack([gl_row, others]))
                edge_attr.append(self._axis_edge_attr(delta[:, 0], "x"))
                edge_index.append(torch.stack([gl_row, others]))
                edge_attr.append(self._axis_edge_attr(delta[:, 1], "y"))

            data_list.append(Data(
                x=x, edge_index=torch.cat(edge_index, dim=1), edge_attr=torch.cat(edge_attr, dim=0)))
        return Batch.from_data_list(data_list)

    # ---- forward -------------------------------------------------------------
    def forward(self, graph_flat: torch.Tensor) -> torch.Tensor:
        """graph_flat [B, N*6] or [B, N, 6] -> graph embedding [B, 128]."""
        if graph_flat.dim() == 2:
            graph_flat = graph_flat.view(-1, self.num_nodes, NODE_FIELDS)
        object_id = graph_flat[..., 0].long()
        active = graph_flat[..., 1]
        is_goal = graph_flat[..., 2]
        xyz = graph_flat[..., 3:6].float()

        node_h = self._node_features(object_id, active, is_goal, xyz)
        batch = self._build_batch(node_h, active, is_goal, xyz)

        h, ei, ea, bidx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        h = self.norm1(F.relu(self.gat1(h, ei, ea)) + h)
        h = self.norm2(F.relu(self.gat2(h, ei, ea)) + h)

        mean_pool = global_mean_pool(h, bidx)
        max_pool = global_max_pool(h, bidx)
        alpha = softmax(self.attn_score(h).squeeze(-1), bidx)
        attn_pool = global_add_pool(alpha.unsqueeze(-1) * h, bidx)
        return self.readout(torch.cat([mean_pool, max_pool, attn_pool], dim=-1))


if __name__ == "__main__":
    # Self-contained smoke test: random graph_flat -> embedding, check backprop.
    torch.manual_seed(0)
    B, V = 4, 300
    enc = MetricGraphEncoder(vocab_size=V, include_node_metric=True)
    g = torch.zeros(B, NUM_NODES, NODE_FIELDS)
    for b in range(B):
        n_active = torch.randint(6, NUM_NODES + 1, (1,)).item()
        g[b, :n_active, 0] = torch.randint(0, V, (n_active,)).float()  # object_id
        g[b, :n_active, 1] = 1.0                                       # active
        g[b, 0, 2] = 1.0                                              # is_goal -> node 0
        g[b, :n_active, 3:6] = torch.randn(n_active, 3) * 4.0         # xyz (floor,floor,height)
    out = enc(g.view(B, -1))
    print("graph_flat", tuple(g.shape), "-> embedding", tuple(out.shape))
    out.sum().backward()
    grad_ok = all(p.grad is not None for p in enc.node_mlp.parameters())
    print("embedding norm/sample:", out.norm(dim=-1).tolist())
    print("backprop reaches node_mlp:", grad_ok)
    print("metric params:", sum(p.numel() for p in enc.parameters()))
    flat = MetricGraphEncoder(vocab_size=V, include_node_metric=False)
    print("flat-non-metric forward:", tuple(flat(g.view(B, -1)).shape))
