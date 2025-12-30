"""
Kolmogorov Detection DSL (Domain Specific Language)

This defines the "language" of programs that can describe objects.
The key insight: simpler programs = more fundamental objects.

Design principles:
1. Compositional: complex shapes from simple primitives
2. Differentiable: can optimize continuous parameters
3. Measurable: program length is well-defined (description length)

Grammar:
    Program := Primitive | Compose(Program, Program) | Transform(Program)
    Primitive := Ellipse | Rectangle | Triangle | Line
    Transform := Translate | Scale | Rotate | Color
    Compose := Union | Subtract | Intersect
"""

from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from enum import Enum
import math


class PrimitiveType(Enum):
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    LINE = "line"
    POLYGON = "polygon"


class ComposeOp(Enum):
    UNION = "union"           # A ∪ B
    SUBTRACT = "subtract"     # A - B
    INTERSECT = "intersect"   # A ∩ B
    OVER = "over"             # A composited over B (with alpha)


@dataclass
class Color:
    """RGB color with alpha."""
    r: float  # 0-1
    g: float  # 0-1
    b: float  # 0-1
    a: float = 1.0  # alpha, 0-1

    def description_length(self) -> float:
        """Bits needed to describe this color."""
        # Quantize to 8-bit per channel = 24 bits for RGB
        # Alpha adds 8 more bits if not 1.0
        bits = 24.0
        if self.a < 1.0:
            bits += 8.0
        return bits

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)


@dataclass
class Transform2D:
    """2D affine transformation."""
    tx: float = 0.0      # translation x
    ty: float = 0.0      # translation y
    sx: float = 1.0      # scale x
    sy: float = 1.0      # scale y
    rotation: float = 0.0  # radians

    def description_length(self) -> float:
        """Bits needed to describe this transform."""
        bits = 0.0
        # Each non-identity parameter costs bits
        # Using ~8 bits per float parameter
        if abs(self.tx) > 1e-6: bits += 8.0
        if abs(self.ty) > 1e-6: bits += 8.0
        if abs(self.sx - 1.0) > 1e-6: bits += 8.0
        if abs(self.sy - 1.0) > 1e-6: bits += 8.0
        if abs(self.rotation) > 1e-6: bits += 8.0
        return bits

    def to_matrix(self) -> 'np.ndarray':
        """Convert to 3x3 affine matrix."""
        import numpy as np
        cos_r = math.cos(self.rotation)
        sin_r = math.sin(self.rotation)
        # Scale -> Rotate -> Translate
        return np.array([
            [self.sx * cos_r, -self.sy * sin_r, self.tx],
            [self.sx * sin_r,  self.sy * cos_r, self.ty],
            [0, 0, 1]
        ])


@dataclass
class Primitive:
    """Base class for geometric primitives."""
    ptype: PrimitiveType
    color: Color
    transform: Transform2D

    # Primitive-specific parameters stored in params dict
    params: dict

    def description_length(self) -> float:
        """Total bits to describe this primitive."""
        # Type identifier: log2(num_primitives) ≈ 3 bits
        bits = 3.0
        # Color
        bits += self.color.description_length()
        # Transform
        bits += self.transform.description_length()
        # Primitive-specific params (8 bits each)
        bits += len(self.params) * 8.0
        return bits


def ellipse(cx: float = 0.5, cy: float = 0.5,
            rx: float = 0.25, ry: float = 0.25,
            color: Color = None, transform: Transform2D = None) -> Primitive:
    """Create an ellipse primitive."""
    return Primitive(
        ptype=PrimitiveType.ELLIPSE,
        color=color or Color(1.0, 1.0, 1.0),
        transform=transform or Transform2D(),
        params={'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry}
    )


def rectangle(x: float = 0.25, y: float = 0.25,
              w: float = 0.5, h: float = 0.5,
              color: Color = None, transform: Transform2D = None) -> Primitive:
    """Create a rectangle primitive."""
    return Primitive(
        ptype=PrimitiveType.RECTANGLE,
        color=color or Color(1.0, 1.0, 1.0),
        transform=transform or Transform2D(),
        params={'x': x, 'y': y, 'w': w, 'h': h}
    )


def triangle(x1: float = 0.5, y1: float = 0.25,
             x2: float = 0.25, y2: float = 0.75,
             x3: float = 0.75, y3: float = 0.75,
             color: Color = None, transform: Transform2D = None) -> Primitive:
    """Create a triangle primitive."""
    return Primitive(
        ptype=PrimitiveType.TRIANGLE,
        color=color or Color(1.0, 1.0, 1.0),
        transform=transform or Transform2D(),
        params={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'x3': x3, 'y3': y3}
    )


def line(x1: float = 0.25, y1: float = 0.25,
         x2: float = 0.75, y2: float = 0.75,
         width: float = 0.02,
         color: Color = None, transform: Transform2D = None) -> Primitive:
    """Create a line primitive."""
    return Primitive(
        ptype=PrimitiveType.LINE,
        color=color or Color(1.0, 1.0, 1.0),
        transform=transform or Transform2D(),
        params={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'width': width}
    )


@dataclass
class Program:
    """
    A program in our DSL that describes a visual object.

    Programs are trees where:
    - Leaves are Primitives
    - Internal nodes are Compose operations

    The description length of a program is the sum of:
    - All primitive description lengths
    - Compose operation costs (2 bits each)
    """

    # Either a single primitive or a composition
    primitive: Optional[Primitive] = None

    # For composite programs
    op: Optional[ComposeOp] = None
    left: Optional['Program'] = None
    right: Optional['Program'] = None

    def is_primitive(self) -> bool:
        return self.primitive is not None

    def description_length(self) -> float:
        """
        Kolmogorov complexity approximation: total bits to describe program.

        This is the key metric - shorter programs are "better" explanations.
        """
        if self.is_primitive():
            return self.primitive.description_length()
        else:
            # Compose op: 2 bits (4 operations)
            op_bits = 2.0
            return op_bits + self.left.description_length() + self.right.description_length()

    def num_primitives(self) -> int:
        """Count total primitives in program."""
        if self.is_primitive():
            return 1
        return self.left.num_primitives() + self.right.num_primitives()

    def depth(self) -> int:
        """Tree depth of program."""
        if self.is_primitive():
            return 1
        return 1 + max(self.left.depth(), self.right.depth())

    def to_string(self, indent: int = 0) -> str:
        """Human-readable program representation."""
        prefix = "  " * indent
        if self.is_primitive():
            p = self.primitive
            params = ", ".join(f"{k}={v:.2f}" for k, v in p.params.items())
            color = f"rgb({p.color.r:.1f},{p.color.g:.1f},{p.color.b:.1f})"
            return f"{prefix}{p.ptype.value}({params}, {color})"
        else:
            left_str = self.left.to_string(indent + 1)
            right_str = self.right.to_string(indent + 1)
            return f"{prefix}{self.op.value}(\n{left_str},\n{right_str}\n{prefix})"


def compose(left: Program, right: Program, op: ComposeOp = ComposeOp.OVER) -> Program:
    """Compose two programs."""
    return Program(op=op, left=left, right=right)


def prim(p: Primitive) -> Program:
    """Wrap a primitive as a program."""
    return Program(primitive=p)


# =============================================================================
# PROGRAM LIBRARY: Common patterns that reduce description length
# =============================================================================

class ProgramLibrary:
    """
    Library of common patterns.

    Key insight: If a pattern appears often, we can assign it a short code.
    This is how we get actual compression - frequent patterns become cheap.

    This is similar to how LZ77/DEFLATE work, but for visual programs.
    """

    def __init__(self):
        # Pattern -> (short_code, canonical_program)
        self.patterns = {}
        self.next_code = 0

    def add_pattern(self, name: str, program: Program) -> int:
        """Add a pattern to the library, return its code."""
        code = self.next_code
        self.patterns[name] = (code, program)
        self.next_code += 1
        return code

    def lookup_cost(self, name: str) -> float:
        """Cost to reference a library pattern."""
        if name not in self.patterns:
            return float('inf')
        # Cost = log2(library_size) to select pattern
        import math
        return math.log2(max(1, len(self.patterns)))

    def instantiate(self, name: str, transform: Transform2D = None) -> Program:
        """Get a program from the library with optional transform."""
        if name not in self.patterns:
            raise ValueError(f"Unknown pattern: {name}")
        _, program = self.patterns[name]
        # TODO: Apply transform to all primitives in program
        return program


# Default library with common shapes
DEFAULT_LIBRARY = ProgramLibrary()

# A car is: body rectangle + 2 wheels + windows
# But we don't pre-define this - it should be DISCOVERED through compression
