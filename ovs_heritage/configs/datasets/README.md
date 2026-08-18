# Dataset schemas

`heritage_facades_v2.py` exports metadata for the v2 two-map contract: stored
`Y_main` uses stable semantic IDs while stored `Y_ornament` uses binary
`0/1/255`. It defines 11 canonical main channels and one independent ornament
channel, but does not implement a model. Existing MMSeg facade configs remain
explicit legacy-v1 single-mask configurations and must not be used to infer or
migrate v2 overlaps.
