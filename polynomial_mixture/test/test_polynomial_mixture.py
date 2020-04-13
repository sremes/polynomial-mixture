"""Unit tests for polynomial_mixture."""
# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

import numpy as np
import pytest

import polynomial_mixture.polynomial_mixture as pm


@pytest.fixture
def simple_data():
    """Return a data generator function."""

    def create_data(polynomial_degree):
        coeffs = np.array([[1.0] + [0.1] * polynomial_degree]).T
        X = np.vstack([np.linspace(-10, 10) ** k for k in range(polynomial_degree + 1)]).T
        y = np.matmul(X, coeffs)
        return X, y

    return create_data


@pytest.mark.parametrize("num_components,polynomial_degree", [(3, 3), (7, 4), (3, 8)])
def test_polynomial_mixture_smoke(simple_data, num_components, polynomial_degree):
    """Smoke test that the output sample from the model is correct."""
    mixture = pm.BayesianPolynomialMixture(num_components=num_components, polynomial_degree=polynomial_degree)
    X, y = simple_data(polynomial_degree)
    model = mixture.create_model(X)
    sample = model.sample()
    assert sample["mixture_probs"].shape == num_components
    assert sample["coefficients"].shape == (num_components, polynomial_degree + 1)
    assert sample["mixture"].shape == y.shape
