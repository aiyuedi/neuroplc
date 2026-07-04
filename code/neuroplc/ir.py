#!/usr/bin/env python3
"""
NeuroPLC — Intermediate Representation (IR) Graph
=====================================================
Compiler IR: a minimal DAG that bridges PyTorch models and SCL code.

Design principles (from GAP-REPORT):
    1. Only 6 operation types — no premature generality
    2. Node IDs are integers — easy to serialize, easy to debug
    3. In-memory graph for transformations, JSON for persistence
    4. validate() catches topology errors before code generation

Operation types:
    MatMul       — W·x + b  (nn.Linear)
    BsplineLUT   — B-spline activation via lookup table (KAN)
    StandardAct  — ReLU / Sigmoid / Tanh / SiLU
    Softmax      — exp normalization
    Argmax       — index of max
    Add          — element-wise addition (KAN base + spline merge)

Edge semantics:
    edge[0] = primary input (data flow)
    edge[1] = secondary input (used by Add for KAN's base+spline merge)

Usage:
    from neuroplc.ir import IRGraph, IRNode, IROpType

    g = IRGraph(name="kan_inference")
    n1 = g.add_node(IROpType.MatMul, name="fc1",
                     attrs={"W": np.array(...), "b": np.array(...)})
    n2 = g.add_node(IROpType.BsplineLUT, name="spline1",
                     attrs={"table": np.array(...), "grid": np.array(...)})
    g.add_edge(n1, n2)  # MatMul output → BsplineLUT input

    g.validate()  # throws if topology is broken
    print(g.to_json())  # serialize for debugging
"""

from __future__ import annotations

import json
import enum
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field

import numpy as np


# ============================================================================
# Operation Type Enum
# ============================================================================

class IROpType(enum.Enum):
    """The 6 operations NeuroPLC compiler supports."""
    MatMul = "matmul"           # W·x + b
    BsplineLUT = "bspline_lut"  # B-spline lookup table + linear interpolation
    StandardAct = "standard_act"  # ReLU / Sigmoid / Tanh / SiLU
    Softmax = "softmax"         # exp normalization
    Argmax = "argmax"           # index of max
    Add = "add"                 # element-wise addition


# ============================================================================
# IR Graph Node
# ============================================================================

@dataclass
class IRNode:
    """
    A single operation in the computation graph.

    Attributes:
        id:        unique integer ID within the graph
        op:        operation type (IROpType)
        name:      human-readable label (e.g. "layer0_matmul")
        attrs:     operation-specific parameters
        inputs:    list of node IDs that feed into this node
        outputs:   list of node IDs that this node feeds into
        shape_in:  expected input shape (for validation, optional)
        shape_out: expected output shape (for validation, optional)
    """
    id: int
    op: IROpType
    name: str = ""
    attrs: dict = field(default_factory=dict)
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    shape_in: Optional[tuple] = None
    shape_out: Optional[tuple] = None

    # ── Attribute accessors (type-safe) ──

    def get_attr(self, key: str, default: Any = None) -> Any:
        """Safe getter with default."""
        return self.attrs.get(key, default)

    def set_attr(self, key: str, value: Any):
        """Set an attribute."""
        self.attrs[key] = value

    # ── Quick type checks ──

    @property
    def is_linear(self) -> bool:
        return self.op == IROpType.MatMul

    @property
    def is_bspline(self) -> bool:
        return self.op == IROpType.BsplineLUT

    @property
    def is_activation(self) -> bool:
        return self.op in (IROpType.BsplineLUT, IROpType.StandardAct)

    @property
    def is_output(self) -> bool:
        return self.op in (IROpType.Softmax, IROpType.Argmax)

    # ── Serialization helpers ──

    def to_dict(self) -> dict:
        """Serialize to dict (for JSON export)."""
        d = {"id": self.id, "op": self.op.value, "name": self.name,
             "inputs": self.inputs, "outputs": self.outputs}
        # Filter out non-serializable attrs
        serializable = {}
        for k, v in self.attrs.items():
            if isinstance(v, np.ndarray):
                serializable[k] = {
                    "__type": "ndarray",
                    "dtype": str(v.dtype),
                    "shape": list(v.shape),
                    "data": v.flatten().tolist()[:100],  # first 100 for preview
                    "truncated": len(v.flatten()) > 100,
                }
            elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
                serializable[k] = v
            else:
                serializable[k] = str(v)
        d["attrs"] = serializable
        if self.shape_in:
            d["shape_in"] = list(self.shape_in)
        if self.shape_out:
            d["shape_out"] = list(self.shape_out)
        return d

    def __repr__(self) -> str:
        in_str = f"←{self.inputs}" if self.inputs else ""
        out_str = f"→{self.outputs}" if self.outputs else ""
        return (f"IRNode(id={self.id}, op={self.op.value}, "
                f"name='{self.name}'{in_str}{out_str})")


# ============================================================================
# IR Graph
# ============================================================================

class IRGraph:
    """
    A directed acyclic graph (DAG) of IRNodes.

    This is the central data structure of the NeuroPLC compiler.
    All transformations (frontend, optimizer, backend) operate on IRGraph.

    Usage:
        g = IRGraph(name="kan_28_16_4")
        n0 = g.add_node(IROpType.MatMul, name="layer0_linear", attrs={...})
        n1 = g.add_node(IROpType.BsplineLUT, name="layer0_bspline", attrs={...})
        n2 = g.add_node(IROpType.Add, name="layer0_merge")
        g.add_edge(n0, n2, port=0)  # linear → merge (primary)
        g.add_edge(n1, n2, port=1)  # bspline → merge (secondary)
        g.validate()

    Edge semantics:
        The 'port' argument in add_edge determines edge ordering in inputs[].
        port=0 = primary data flow (linear output → next layer)
        port=1 = secondary (used by Add nodes for KAN merge)
    """

    def __init__(self, name: str = "ir_graph"):
        self.name = name
        self.nodes: dict[int, IRNode] = {}
        self._next_id: int = 0
        self._input_node_id: Optional[int] = None
        self._output_node_id: Optional[int] = None

    # ── Node management ──

    def add_node(self, op: IROpType, name: str = "",
                 attrs: dict = None,
                 shape_in: Optional[tuple] = None,
                 shape_out: Optional[tuple] = None) -> IRNode:
        """Add a node to the graph and return it."""
        node = IRNode(
            id=self._next_id,
            op=op,
            name=name or f"{op.value}_{self._next_id}",
            attrs=attrs or {},
            shape_in=shape_in,
            shape_out=shape_out,
        )
        self.nodes[self._next_id] = node
        self._next_id += 1
        return node

    def get_node(self, node_id: int) -> IRNode:
        """Get a node by ID. Raises KeyError if not found."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not in graph '{self.name}'")
        return self.nodes[node_id]

    def add_edge(self, src: IRNode, dst: IRNode, port: int = 0):
        """Connect src → dst. port controls ordering in dst.inputs."""
        if dst.id not in src.outputs:
            src.outputs.append(dst.id)
        # Ensure dst.inputs has room for this port
        while len(dst.inputs) <= port:
            dst.inputs.append(-1)
        dst.inputs[port] = src.id
        # Clean up -1 placeholders
        dst.inputs = [i for i in dst.inputs if i >= 0]

    # ── Graph properties ──

    @property
    def input_nodes(self) -> list[IRNode]:
        """Nodes with no incoming edges (graph inputs)."""
        return [n for n in self.nodes.values() if not n.inputs]

    @property
    def output_nodes(self) -> list[IRNode]:
        """Nodes with no outgoing edges (graph outputs)."""
        return [n for n in self.nodes.values() if not n.outputs]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def op_counts(self) -> dict[str, int]:
        """Count of each operation type in the graph."""
        counts = {}
        for n in self.nodes.values():
            counts[n.op.value] = counts.get(n.op.value, 0) + 1
        return counts

    # ── Traversal ──

    def topological_order(self) -> list[int]:
        """
        Return node IDs in topological order (inputs first, outputs last).
        Uses Kahn's algorithm. Raises ValueError if graph has a cycle.
        """
        in_degree = {nid: len(node.inputs) for nid, node in self.nodes.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for out_id in self.nodes[nid].outputs:
                if out_id in in_degree:
                    in_degree[out_id] -= 1
                    if in_degree[out_id] == 0:
                        queue.append(out_id)

        if len(order) != len(self.nodes):
            remaining = set(self.nodes.keys()) - set(order)
            raise ValueError(
                f"Graph has a cycle! Nodes not reached: {remaining}. "
                f"Check for circular edges."
            )
        return order

    # ── Validation ──

    def validate(self) -> list[str]:
        """
        Validate graph topology and return list of warnings.
        Empty list = valid.

        Checks:
            1. No cycles (topological sort succeeds)
            2. No dangling edges (all referenced node IDs exist)
            3. Every node reachable from some input
            4. At least one input and one output node
            5. BsplineLUT nodes have grid and table attrs
            6. MatMul nodes have W and b attrs
        """
        warnings = []

        # 1. Check for cycles
        try:
            self.topological_order()
        except ValueError as e:
            warnings.append(str(e))

        # 2. Check edge integrity
        all_ids = set(self.nodes.keys())
        for n in self.nodes.values():
            for in_id in n.inputs:
                if in_id not in all_ids:
                    warnings.append(
                        f"Node {n.id} ('{n.name}') references "
                        f"non-existent input node {in_id}")
            for out_id in n.outputs:
                if out_id not in all_ids:
                    warnings.append(
                        f"Node {n.id} ('{n.name}') references "
                        f"non-existent output node {out_id}")

        # 3. Check reachability
        if self.input_nodes:
            visited = set()
            queue = [n.id for n in self.input_nodes]
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                for out_id in self.nodes[nid].outputs:
                    if out_id not in visited:
                        queue.append(out_id)
            unreachable = all_ids - visited
            if unreachable:
                warnings.append(
                    f"Unreachable nodes (no path from any input): {unreachable}")

        # 4. Graph structure
        if not self.input_nodes:
            warnings.append("Graph has no input nodes (no nodes without inputs)")
        if not self.output_nodes:
            warnings.append("Graph has no output nodes (all nodes feed somewhere)")

        # 5. Operation-specific checks
        for n in self.nodes.values():
            if n.op == IROpType.BsplineLUT:
                if "grid" not in n.attrs:
                    warnings.append(
                        f"BsplineLUT node {n.id} ('{n.name}') missing 'grid' attr")
                if "table" not in n.attrs:
                    warnings.append(
                        f"BsplineLUT node {n.id} ('{n.name}') missing 'table' attr")
            elif n.op == IROpType.MatMul:
                if "W" not in n.attrs:
                    warnings.append(
                        f"MatMul node {n.id} ('{n.name}') missing 'W' attr")
                if "b" not in n.attrs:
                    warnings.append(
                        f"MatMul node {n.id} ('{n.name}') missing 'b' attr")

        return warnings

    @property
    def is_valid(self) -> bool:
        """Quick check: is the graph valid? (no errors)"""
        return len(self.validate()) == 0

    # ── Serialization ──

    def to_dict(self) -> dict:
        """Serialize entire graph to a plain dict."""
        return {
            "name": self.name,
            "node_count": self.node_count,
            "op_counts": self.op_counts,
            "nodes": [n.to_dict() for n in self.nodes.values()],
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Serialize to JSON string. If path is given, write to file.

        Returns:
            JSON string representation.
        """
        d = self.to_dict()

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        json_str = json.dumps(d, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str

    # ── Display ──

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        lines = [
            f"IRGraph '{self.name}' — {self.node_count} nodes, "
            f"{len(self.input_nodes)} inputs, {len(self.output_nodes)} outputs",
            f"Operations: {self.op_counts}",
            "Nodes (topological order):",
        ]
        try:
            order = self.topological_order()
        except ValueError:
            order = list(self.nodes.keys())
        for nid in order:
            n = self.nodes[nid]
            in_str = f" ← [{', '.join(map(str, n.inputs))}]" if n.inputs else ""
            lines.append(f"  [{n.id}] {n.op.value:15s} {n.name}{in_str}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"IRGraph('{self.name}', {self.node_count} nodes)"

    def __len__(self) -> int:
        return self.node_count


# ============================================================================
# Quick construction helpers
# ============================================================================

def build_mlp_ir(layer_dims: list[int], activation: str = "relu",
                 graph_name: str = "mlp") -> IRGraph:
    """
    Build an MLP IR graph from a list of layer dimensions.
    This is a convenience function for testing and quick prototyping.

    Args:
        layer_dims: [input_dim, hidden1, hidden2, ..., output_dim]
        activation: "relu" | "sigmoid" | "tanh"
        graph_name: label for the IR graph

    Returns:
        IRGraph with MatMul + StandardAct nodes for each hidden layer,
        and a final MatMul + Softmax for the output.

    Raises:
        ValueError: if weights don't exist yet (use frontend.py for real models)

    Note:
        This creates PLACEHOLDER weights (all zeros).
        For real compilation, use neuroplc.frontend.mlp_to_ir().
    """
    g = IRGraph(name=graph_name)

    prev_out_dim = layer_dims[0]
    # Placeholder input node (no W/b — real weights come from frontend.py)
    last_node = g.add_node(IROpType.MatMul, name="input_placeholder",
                           attrs={"W": np.eye(layer_dims[0], dtype=np.float32),
                                  "b": np.zeros(layer_dims[0], dtype=np.float32)},
                           shape_in=(layer_dims[0],), shape_out=(layer_dims[0],))

    for i in range(1, len(layer_dims)):
        in_dim = prev_out_dim
        out_dim = layer_dims[i]
        is_last = (i == len(layer_dims) - 1)

        # MatMul
        fc = g.add_node(
            IROpType.MatMul,
            name=f"fc{i}",
            attrs={
                "W": np.zeros((out_dim, in_dim), dtype=np.float32),
                "b": np.zeros((out_dim,), dtype=np.float32),
            },
            shape_in=(in_dim,),
            shape_out=(out_dim,),
        )
        g.add_edge(last_node, fc)

        if is_last:
            # Output: Softmax
            sm = g.add_node(IROpType.Softmax, name="softmax",
                            shape_in=(out_dim,), shape_out=(out_dim,))
            g.add_edge(fc, sm)
            last_node = sm
        else:
            # Hidden: Activation
            act = g.add_node(
                IROpType.StandardAct,
                name=f"{activation}{i}",
                attrs={"type": activation},
                shape_in=(out_dim,),
                shape_out=(out_dim,),
            )
            g.add_edge(fc, act)
            last_node = act

        prev_out_dim = out_dim

    return g


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    print("NeuroPLC IR — Sanity Check\n")

    # Build a minimal KAN-style IR graph
    g = IRGraph(name="kan_simple")

    # Layer 0: 28→16
    linear0 = g.add_node(IROpType.MatMul, name="l0_linear",
        attrs={"W": np.zeros((16, 28), dtype=np.float32),
               "b": np.zeros((16,), dtype=np.float32)})
    bspline0 = g.add_node(IROpType.BsplineLUT, name="l0_bspline",
        attrs={"grid": np.linspace(-3, 3, 20),
               "table": np.random.randn(20).astype(np.float32) * 0.1})
    merge0 = g.add_node(IROpType.Add, name="l0_merge")

    g.add_edge(linear0, merge0, port=0)
    g.add_edge(bspline0, merge0, port=1)

    # Layer 1: 16→4
    linear1 = g.add_node(IROpType.MatMul, name="l1_linear",
        attrs={"W": np.zeros((4, 16), dtype=np.float32),
               "b": np.zeros((4,), dtype=np.float32)})
    bspline1 = g.add_node(IROpType.BsplineLUT, name="l1_bspline",
        attrs={"grid": np.linspace(-3, 3, 20),
               "table": np.random.randn(20).astype(np.float32) * 0.1})
    merge1 = g.add_node(IROpType.Add, name="l1_merge")

    g.add_edge(merge0, linear1, port=0)
    g.add_edge(merge0, bspline1, port=0)
    g.add_edge(linear1, merge1, port=0)
    g.add_edge(bspline1, merge1, port=1)

    # Output
    softmax = g.add_node(IROpType.Softmax, name="softmax")
    argmax = g.add_node(IROpType.Argmax, name="argmax")
    g.add_edge(merge1, softmax)
    g.add_edge(softmax, argmax)

    print(g.summary())
    print()

    # Validate
    warnings = g.validate()
    if warnings:
        print(f"Validation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("✅ Graph is valid (no warnings)")

    # Serialize
    json_str = g.to_json()
    print(f"\nSerialized: {len(json_str)} chars")
    print(f"Operations: {g.op_counts}")

    # Test MLP builder
    print("\n--- MLP IR ---")
    mlp_g = build_mlp_ir([28, 32, 16, 4])
    print(mlp_g.summary())
    print(f"Valid: {mlp_g.is_valid}")
