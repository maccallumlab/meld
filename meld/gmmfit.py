"""
Routines to build GMMDistanceRestraints from data.
"""

import math

import numpy as np  # type: ignore
from scipy import stats  # type: ignore
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture  # type: ignore
from sklearn.model_selection import KFold, RandomizedSearchCV  # type: ignore

from meld.system import restraints


def fit_gmm(
    max_components,
    n_distances,
    atoms,
    distances,
    regularization_type="bic",
    covariance_type="diag",
):
    """
    Fit a GMM to a set of distances.

    This routine will fit a Gaussian mixture model from a set
    of input distances using sklearn_. The resulting set of parameters can
    be used to initialize a `GMMDistanceRestraint` in a MELD simulation.

    .. _sklearn: http://scikit-learn.org/stable/modules/mixture.html

    Parameters
    ----------
    max_components: int
        Maximum number of components to use in fitting GMM.
    n_distances: int
        Number of distances involved in GMM
    atoms: list of (int, str, int, str) tuples.
        The atoms that are involved in each distance are specified
        as a list of `n_distances` tuples, each of the form
        (r1, n1, r2, n2), where r1, r2 are the integer residue
        indices starting from one, and n1, n2 are the atom names.
    distances: array_like(n_dim=2)
        An (n_samples, n_distances) array of distances (in nm) to fit.
    regularization_type: str
        The type of regularization to use, options are "bic"
        and "dirichlet".
    covariance_type: str
        The form of the covariance matrix, options are "diag"
        and "full".

    Returns
    -------
    GMMParams
        The fit parameters, which can be used to initialize
        a `meld.system.restraints.GMMDistanceRestraint` using
        ``GMMDistanceRestraint.from_params``.

    Notes
    -----
    There are two ways to regularize in order to prevent over fitting.

    ``regularization_type="bic"`` will use the Bayesian information
    criterion to penalize models that have more parameters. When
    using ``bic``, The final number of components in the model
    will be less than or equal to `max_components`.

    ``regularization_type=dirichlet`` will use a Dirichlet process
    prior on the weight distributions. The final number of components
    in the model will always be equal to `max_components`, but most
    of the weights will be small.

    There are two forms for the covariance matrix, which differ in
    the number of parameters and expressiveness.

    ``covariance_type="diag"`` will fit using a diagonal covariance
    matrix. This has few parameters, but does not capture correlations
    between input distances. Typically, choosing ``"diag"`` will
    result in a model with more components.

    ``covariance_type="full"`` will fit using a full representation
    of the covariance matrix. This captures correlations between
    input distances, but has far more parameters and is potentially
    prone to over fitting.
    """

    #
    # Constants
    #
    N_INIT = 25
    MAX_ITER = 1000
    KFOLD_SPLITS = 5
    REG_COVAR = 1e-4
    RANDOMSEARCH_TRIALS = 32

    #
    # Check the inputs
    #
    if distances.shape[1] != n_distances:
        raise ValueError("distances must have shape (n_samples, n_distances)")

    if len(atoms) != n_distances:
        raise ValueError(
            "atoms must be a list of (ind1, name1, ind2, name2) of "
            "length n_components"
        )

    if regularization_type not in ["bic", "dirichlet"]:
        raise ValueError('regularization_type must be one of ["bic", "dirichlet"]')

    if covariance_type not in ["diag", "full"]:
        raise ValueError('covariance_type must be one of ["diag", "full"]')

    if max_components < 1:
        raise ValueError("max_components must be >= 1")
    if max_components > 32:
        raise ValueError("MELD supports a maximum of 32 GMM components")

    #
    # Create and fit the model
    #
    if regularization_type == "bic":
        # BIC fit
        # Search different values of n_components to find the minimal
        # BIC.
        models = []
        for i in range(1, max_components + 1):
            g = GaussianMixture(
                n_components=i,
                n_init=N_INIT,
                max_iter=MAX_ITER,
                covariance_type=covariance_type,
                reg_covar=REG_COVAR,
            )
            g.fit(distances)
            models.append((g.bic(distances), g))

        gmm = sorted(models, key=lambda x: x[0])[0][1]

    else:
        # Dirichlet process fit
        # use RandomSearchCV to optimize hyperparameters
        params = {
            "weight_concentration_prior": LogUniformSampler(1e-6, 10),
            "mean_precision_prior": LogUniformSampler(1, 10),
        }
        model = BayesianGaussianMixture(
            max_components,
            n_init=N_INIT,
            max_iter=MAX_ITER,
            covariance_type=covariance_type,
            reg_covar=REG_COVAR,
        )
        rs = RandomizedSearchCV(
            model,
            param_distributions=params,
            n_iter=RANDOMSEARCH_TRIALS,
            cv=KFold(n_splits=KFOLD_SPLITS, shuffle=True),
        )
        rs.fit(distances)
        gmm = rs.best_estimator_

    # turn the vector representation of the diagonal into a full
    # precision matrix
    if covariance_type == "diag":
        precisions = gmm.precisions_
        assert len(precisions.shape) == 2
        new_precisions = []
        for i in range(precisions.shape[0]):
            new_precisions.append(np.diag(precisions[i, :]))
        precisions = np.array(new_precisions)
    else:
        precisions = gmm.precisions_

    # convert the list of atoms into the correct form
    new_atoms = []
    for r1, n1, r2, n2 in atoms:
        new_atoms.append((r1, n1))
        new_atoms.append((r2, n2))

    # Return the parameters for a GMM
    return restraints.GMMParams(
        n_components=gmm.weights_.shape[0],
        n_distances=n_distances,
        atoms=new_atoms,
        weights=gmm.weights_,
        means=gmm.means_,
        precisions=precisions,
    )


class LogUniformSampler:
    """Sample uniformly in log space"""

    def __init__(self, min_, max_):
        assert min_ > 0
        assert max_ > 0
        self.log_min = math.log(min_)
        self.log_max = math.log(max_)
        self.uniform = stats.uniform(
            loc=self.log_min, scale=self.log_max - self.log_min
        )

    def rvs(self, random_state):
        return math.exp(self.uniform.rvs(random_state=random_state))
