"""A modelling step guarded by ``try``/``except``.

Stresses: statements whose execution is conditional on a runtime exception.  The
except branch rebinds the same name, so versioning must not make one arm
unreachable.
"""
import cadquery as cq

body_diameter = 26.0
body_height = 18.0
flange_diameter = 38.0
flange_thickness = 4.0
aggressive_fillet = 40.0
safe_fillet = 1.5
bore_diameter = 10.0

body = cq.Workplane("XY").circle(body_diameter / 2).extrude(body_height)
flange = cq.Workplane("XY").circle(flange_diameter / 2).extrude(flange_thickness)
joined = body.union(flange)

try:
    # Deliberately too large: OpenCascade rejects it and the fallback runs.
    finished = joined.edges(">Z").fillet(aggressive_fillet)
except Exception:
    finished = joined.edges(">Z").fillet(safe_fillet)

result = finished.faces(">Z").workplane().circle(bore_diameter / 2).cutThruAll()
