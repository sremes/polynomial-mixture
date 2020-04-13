"""Bayesian polynomial mixture model."""
# pylint: disable=invalid-name

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class BayesianPolynomialMixture:  # pylint: disable=too-few-public-methods
    """Handles creation of a polynomial mixture model."""

    def __init__(self, num_components=5, polynomial_degree=3):
        """Creates polynomial mixture with given mixture components of given degree."""
        self.num_components = num_components
        self.polynomial_degree = polynomial_degree
        self.coefficient_precisions = [10.0 ** x for x in range(self.polynomial_degree + 1)]
        self.concentration = np.array([0.1 for _ in range(self.num_components)])
        self.wishart_df = self.polynomial_degree + 2.0
        self.student_df = 2

    def create_model(self, X):
        """Defines the joint distribution of the mixture model."""
        precision_scale = np.repeat(np.expand_dims(self.coefficient_precisions, 0), self.num_components, axis=0)
        joint_distribution = tfd.JointDistributionNamed(
            dict(
                precision=tfd.Independent(
                    tfd.WishartLinearOperator(
                        df=self.wishart_df,
                        scale=tf.linalg.LinearOperatorDiag(precision_scale),
                        input_output_cholesky=True,
                        name="precision",
                    ),
                    reinterpreted_batch_ndims=1,
                ),
                coefficients=lambda precision: tfd.Independent(
                    tfd.MultivariateNormalTriL(
                        loc=0, scale_tril=tfb.MatrixInverseTriL()(precision), name="coefficients"
                    ),
                    reinterpreted_batch_ndims=1,
                ),
                scale=tfd.HalfCauchy(loc=np.float64(0.0), scale=np.float64(1.0), name="noise_scale"),
                mixture_probs=tfd.Dirichlet(concentration=self.concentration, name="mixture_probs"),
                mixture=lambda mixture_probs, coefficients, scale: tfd.Sample(
                    tfd.MixtureSameFamily(
                        mixture_distribution=tfd.Categorical(probs=mixture_probs, name="mixture_distribution"),
                        components_distribution=tfd.StudentT(
                            df=self.student_df,
                            loc=tf.linalg.matmul(X, coefficients, transpose_b=True),
                            scale=scale,
                            name="sample_likelihood",
                        ),
                        name="mixture_components",
                    ),
                    sample_shape=1,
                ),
            ),
            name="joint_distribution",
        )
        return joint_distribution
