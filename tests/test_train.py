import json
import os
import joblib
import pytest
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from src.train import load_config, train_model

def test_config_file_loading():
    config = load_config("config/config.json")
    assert isinstance(config, dict)
    assert "C" in config
    assert "solver" in config
    assert "max_iter" in config
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_training():
    config = load_config("config/config.json")
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    assert hasattr(model, "classes_")

def test_model_accuracy():
    config = load_config("config/config.json")
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    acc = model.score(X, y)
    assert acc > 0.8
