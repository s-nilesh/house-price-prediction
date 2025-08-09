import pytest 
import numpy as np
from algorithms.linear_regression import LinearRegression


@pytest.fixture
def model():
    """Fixture to initialize the Linear Regression model."""
    return LinearRegression(learning_rate=0.01, epochs=1000)

@pytest.fixture
def simple_data():
    """
    Simple fixture for generating linear data (y = 2x + 1).
    """
    X_train = np.array([[1], [2], [3], [4], [5]])
    y_train = np.array([[3], [5], [7], [9], [11]])
    return X_train, y_train

def test_initialization(model, simple_data):
    """Test initialization of the model."""
    # Check if coefficients are initialized correctly (random values between 0 and 1)
    X_train, y_train = simple_data
    model.fit(X_train, y_train)
    assert model.coefs.shape == (2, 1), f"Expected coefficients shape (2, 1), got {model.coefs.shape}"
    assert np.all(model.coefs != 0), "Coefficients should not be zero"

def test_fit_method(model, simple_data):
    """Test if the model can successfully fit the data."""
    X_train, y_train = simple_data
    model.fit(X_train, y_train)
    
    # After fitting, check if the coefficients are updated (they should not be zero)
    assert np.all(model.coefs != 0), "Coefficients should not remain zero after training"
    
    # Check if the model has learned a meaningful relationship (slope should be close to 2)
    assert np.isclose(model.coefs[1], 2, atol=0.1), f"Slope is not close to 2, got {model.coefs[1]}"
    assert np.isclose(model.coefs[0], 1, atol=0.1), f"Intercept is not close to 1, got {model.coefs[0]}"

def test_predict_method(model, simple_data):
    """Test if the model makes correct predictions."""
    X_train, y_train = simple_data
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_train)
    
    # Check if the predicted shape matches the input shape
    assert y_pred.shape == y_train.shape, f"Expected shape {y_train.shape}, got {y_pred.shape}"
    
    # Test if the predictions are reasonably close to the expected values
    np.testing.assert_allclose(y_pred, y_train, atol=1e-1, err_msg="Predictions are not close enough to the actual values")

def test_cost_function(model, simple_data):
    """Test if the cost function (MSE) is computed correctly."""
    X_train, y_train = simple_data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    
    cost = model.cost_function(y_pred, y_train)
    
    # The cost should be a scalar value and should be a positive number
    assert isinstance(cost, float), f"Expected float cost value, got {type(cost)}"
    assert cost > 0, f"Cost function returned a non-positive value: {cost}"

def test_edge_case_empty_data(model):
    """Test with empty data."""
    X_empty = np.array([])
    y_empty = np.array([])
    
    with pytest.raises(ValueError):
        model.fit(X_empty, y_empty)

def test_edge_case_constant_data(model):
    """Test with constant data (no variance)."""
    X_constant = np.array([[1], [1], [1], [1], [1]])
    y_constant = np.array([[3], [3], [3], [3], [3]])
    
    model.fit(X_constant, y_constant)
    
    # Predictions should be constant and close to the expected value (3)
    y_pred = model.predict(X_constant)
    np.testing.assert_allclose(y_pred, np.array([[3], [3], [3], [3], [3]]), atol=1e-1)