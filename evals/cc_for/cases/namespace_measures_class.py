"""The Zero-to-CAD ``SimpleNamespace`` + builder-class idiom, nested two deep.

Stresses: namespace flattening has to lift ``m.shell.width`` into a named
parameter while keeping the container alive, because the class reads it through
``self.measures`` at runtime.
"""
import cadquery as cq
from types import SimpleNamespace as Measures


class EnclosureBase:
    def __init__(self, measures):
        self.measures = measures
        self.model = None
        self.build()

    def build(self):
        m = self.measures
        shell = (
            cq.Workplane("XY")
            .rect(m.shell.width, m.shell.depth)
            .extrude(m.shell.height)
        )
        hollow = shell.faces(">Z").shell(-m.shell.wall)
        filleted = hollow.edges("|Z").fillet(m.shell.corner_radius)
        self.model = (
            filleted.faces("<Z")
            .workplane()
            .rarray(m.mount.spacing, m.mount.spacing, 2, 2)
            .hole(m.mount.hole_diameter)
        )


measures = Measures(
    shell=Measures(width=70.0, depth=50.0, height=28.0, wall=2.5, corner_radius=4.0),
    mount=Measures(spacing=40.0, hole_diameter=3.5),
)

result = EnclosureBase(measures).model
