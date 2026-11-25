import pytest
import graphics.terrain_lod_crack_stitch as mod

CASES = [
    ("align_boundary_vertices", (1,)),
    ("lod_hysteresis", (1,)),
    ("clamp_edge_sample", (1,)),
    ("normal_by_lod", (1,)),
    ("pool_by_size_bucket", (1,)),
    ("prebuild_before_switch", (1,)),
    ("load_heightmap", (1,)),
    ("build_chunk", (1,)),
    ("stitch", (1,)),
    ("present", (1,)),
    ("validate_terrain", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
