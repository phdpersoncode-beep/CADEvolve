import cadquery as cq

r_small = 2
r_mid = 25
h1 = 36
h2 = 40
wp = cq.Workplane("XY")
wp = wp.workplane(offset=-h1).circle(r_small)
wp = wp.workplane(offset=h1).circle(r_mid)
wp = wp.workplane(offset=h2).circle(r_mid)
wp = wp.workplane(offset=h1).circle(r_small)
result = wp.loft(combine=True)
