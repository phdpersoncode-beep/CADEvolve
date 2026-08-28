"""A ``for`` whose *iterable* is a CadQuery chain, not a list of numbers.

Stresses: lowering rewrites the loop body and the names in the loop header, but
not a chain sitting in the header itself, so ``part.faces('>Z').vals()`` stays a
fluent expression and the explicit-step contract is broken in the one place
``chains_lowered`` looks.  Nothing is silently wrong -- Python evaluates the
iterable once at loop entry and the solid comes out identical -- which is why
this is recorded as a self-reported coverage gap rather than a geometry defect.

Placement is unaffected: the header's reads anchor the loop's parameters above
it exactly as a plain ``range`` would.
"""
import cadquery as cq

block_size = 24.0
pocket_size = 6.0
pocket_depth = 2.0
edge_fillet = 0.8

block = cq.Workplane("XY").box(block_size, block_size, block_size)

for face_index, face in enumerate(block.faces(">Z").vals()):
    block = (
        block.faces(">Z")
        .workplane()
        .rect(pocket_size, pocket_size)
        .cutBlind(-pocket_depth)
    )

result = block.edges("|Z").fillet(edge_fillet)
