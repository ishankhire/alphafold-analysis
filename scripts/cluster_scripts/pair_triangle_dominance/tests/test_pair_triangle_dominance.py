import math, numpy as np
from pair_triangle_dominance_common import *
def test_margin_and_normalization():
 m,z=channel_measurements(np.array([[[5.],[3.],[1.]]]))
 assert m[0][0,0]==1 and np.isclose(m[2][0,0],1/9)
def test_variants_cancel_only_after_symmetrizing():
 z=np.zeros((3,3,1)); z[2,1]=5; z[1,2]=-5
 s=transform_sides(z,np.array([[2,1,0]])); assert s[2][0,0,0]==2.5 and s[3][0,0,0]==0
def test_sampling_is_deterministic_and_valid():
 a,r,_=sample_triples('x',20); b,_,_=sample_triples('x',20); assert np.array_equal(a,b) and validate_samples(a,20)
