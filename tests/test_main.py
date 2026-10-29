### this file knows what "flat_model" (defined in conftest.py) is, once we import pytest

import pytest
from brightify.main import Flat

def test_model_creation(flat_model):
    assert isinstance(flat_model, Flat)

def test_model_data(flat_model):
    assert len(getattr(flat_model, "data")) == 100

def test_model_particle_type(flat_model):
    assert flat_model.data["Kf"][0] == 2112

def test_model_filtering(flat_model):
    flat_model.apply_filter(energy=(0,1e-10))
    assert len(flat_model.data_filter) == 0
    