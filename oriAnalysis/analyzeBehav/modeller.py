import numpy as np
from scipy.stats import circmean#, circstd, circvar
from scipy.stats import vonmises, uniform
from scipy.optimize import minimize


def circ_stats(data, axis=None):
    """
    Compute circular statistics for a given data array in degrees.
    
    Parameters
    ----------
    data : array_like
        Array of angles in degrees.
    axis : int, optional
        Axis along which the circular statistics are computed. Default is None.
    
    Returns
    -------
    mean : ndarray
        Mean of the circular data.
    std : ndarray
        Standard deviation of the circular data.
    var : ndarray
        Variance of the circular data.
    skew : ndarray
        Skewness of the circular data.
    kurt : ndarray
        Kurtosis of the circular data.
    """
    if axis is None:
        data = np.ravel(data)
        axis = 0

    data = (data * 2) 
    data_rad = np.radians(data)
    low, high = -np.pi, np.pi

    # Compute mean
    mean_rad = circmean(data, low=low, high=high, axis=axis)

    # Compute resultant vector length
    n = data_rad.shape[axis]
    R = np.abs(np.sum(np.exp(1j * data), axis=axis)) / n

    # Circular standard deviation
    std_rad = 0.5 * np.sqrt(-2 * np.log(R))

    # Convert mean and std back to degrees
    mean = np.degrees(mean_rad) / 2
    std = np.degrees(std_rad) / 2

    # Circular variance
    var = 1 - R

    # Compute skewness
    skew = np.mean(np.sin(2 * (data_rad - mean_rad)), axis=axis)

    # Compute kurtosis
    kurt = np.mean(np.cos(2 * (data_rad - mean_rad)), axis=axis)
    
    return mean, std, var, skew, kurt

def pure_vonmises_pdf(x, K):
    """
    Compute the probability density function of a von Mises distribution.

    Parameters
    ----------
    x : array_like
        Array of angles in radians between -pi and pi.
    K : float
        Concentration parameter of the von Mises distribution.

    Returns
    -------
    pdf : ndarray
        Probability density function of the von Mises distribution.
    """
    return vonmises.pdf(x, K, loc=0)

def mixture_model_pdf(x, g, K):
    """
    Compute the probability density function of a mixture model of von Mises and uniform distributions.

    Parameters
    ----------
    x : array_like
        Array of angles in radians between -pi and pi.
    g : float
        Weighting factor for the uniform distribution.
    K : float
        Concentration parameter of the von Mises distribution.

    Returns
    -------
    pdf : ndarray
        Probability density function of the mixture model.
    """
    vonmisses_part = (1-g) * vonmises.pdf(x, K, loc=0)
    uniform_part = g * uniform.pdf(x, loc=-np.pi, scale=2*np.pi)
    return vonmisses_part + uniform_part

def negative_log_likelihood(params, modelName, x):
    """
    Compute the negative log-likelihood of a mixture model of von Mises and uniform distributions.

    Parameters
    ----------
    params : array_like
        Parameters of the mixture model.
    x : array_like
        Array of angles in degrees.

    Returns
    -------
    nll : float
        Negative log-likelihood of the mixture model.
    """
    if modelName == 'vonMises':
        K = params[0]
        pdf = pure_vonmises_pdf(x, K)
    elif modelName == 'mixtureModel':
        g, K = params
        pdf = mixture_model_pdf(x, g, K)
        
    nll = -np.sum(np.log(pdf) + 1e-10) # Add a small number to avoid log(0)
    return nll

def fit_model(x, modelName, max_trials=1000, tol=1e-6):
    '''
    Fit a mixture model of von Mises and uniform distributions to the data.
    '''
    x = np.radians(2*x) # Convert to radians and between -pi and pi

    if modelName == 'vonMises':
        bounds = [(0, None)]
    elif modelName == 'mixtureModel':
        bounds = [(0, 1), (0, None)]
    else:
        raise ValueError(f"Invalid model name: {modelName}")
    
    best_fit = None
    best_nll = np.inf
    no_improvement_iters = 0

    for trial in range(max_trials):
        init_g = np.random.uniform(0, 1)
        init_K = np.exp(np.random.uniform(np.log(1), np.log(100))) # Log-uniform in range [1, 100]
        init_params = [init_g, init_K]

        # Optimize the negative log-likelihood function
        result = minimize(negative_log_likelihood, init_params, args=(modelName, x), bounds=bounds)

        if result.success:
            nll = result.fun

            if nll < best_nll - tol:
                best_fit = result
                best_nll = nll
                no_improvement_iters = 0
            else:
                no_improvement_iters += 1
        
        if no_improvement_iters >= 20:
            break

    if best_fit is not None:
        print(f"Optimization successful after {trial} iterations, NLL = {best_nll}")
        return best_fit.x
    else:
        raise ValueError("Optimization failed to find a suitable solution")
