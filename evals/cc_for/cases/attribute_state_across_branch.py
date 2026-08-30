"""A builder that rebinds ``self.model`` inside an ``if``, then keeps building.

Stresses: attribute targets are invisible to a name-based analysis, so a branch
that writes ``self.model`` looks like a branch that writes nothing.  The alias
recorded before the branch then survives it, and the chamfer after the branch is
applied to the model as it stood *before* the gusset was unioned in.

Nothing raises: the canonical program builds a bracket that is missing its
gusset, which is the kind of loss only a geometry comparison finds.
"""
import cadquery as cq

leg_length = 60.0
leg_height = 50.0
leg_thickness = 8.0
gusset_width = 18.0
gusset_height = 18.0
gusset_thickness = 6.0
edge_chamfer = 1.0
include_gusset = True


class Bracket:
    def __init__(self):
        self.model = (
            cq.Workplane("XY")
            .moveTo(0, 0)
            .lineTo(0, leg_height)
            .lineTo(leg_thickness, leg_height)
            .lineTo(leg_thickness, leg_thickness)
            .lineTo(leg_length, leg_thickness)
            .lineTo(leg_length, 0)
            .close()
            .extrude(leg_thickness)
        )
        if include_gusset:
            gusset = (
                cq.Workplane("XY")
                .moveTo(leg_thickness, leg_thickness)
                .lineTo(leg_thickness + gusset_width, leg_thickness)
                .lineTo(leg_thickness, leg_thickness + gusset_height)
                .close()
                .extrude(gusset_thickness)
            )
            self.model = self.model.union(gusset)
        self.model = self.model.edges("|Z").chamfer(edge_chamfer)


result = Bracket().model
