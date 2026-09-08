import numpy as np
import pytest
from src.tiling import make_tiles

def test_small_image_one_tile():
    tiles=make_tiles(np.zeros((400,512,3),dtype=np.uint8))
    assert len(tiles)==1 and tiles[0].valid_width==512 and tiles[0].valid_height==400

def test_large_image_reaches_edges():
    tiles=make_tiles(np.zeros((1300,1800,3),dtype=np.uint8))
    assert max(t.x+t.valid_width for t in tiles)==1800
    assert max(t.y+t.valid_height for t in tiles)==1300

def test_invalid_overlap():
    with pytest.raises(ValueError): make_tiles(np.zeros((10,10,3)),overlap=1)
