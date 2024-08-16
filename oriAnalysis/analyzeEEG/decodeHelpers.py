import numpy as np
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)


def smooth_data(data, window_size=100):
    smoothed_data = np.zeros_like(data)
    for epoch, chan in product(range(data.shape[0]), range(data.shape[1])):
        smoothed_data[epoch, chan, :] = np.convolve(
                                                    data[epoch, chan, :],
                                                    np.ones(window_size) / window_size, 
                                                    mode='same'
                                            )
    return smoothed_data

def gaussian_smooth_1d(data, sigma=1):
    smoothed_data = np.zeros_like(data)
    from scipy.ndimage import gaussian_filter1d
    for epoch, chan in product(range(data.shape[0]), range(data.shape[1])):
        smoothed_data[epoch, chan, :] = gaussian_filter1d(data[epoch, chan, :], sigma, axis=-1)
    return smoothed_data

def moving_average_downsample(data, t_array, window_size=5):
    downsampled_length = (data.shape[2] - window_size + 1) // window_size
    downsampled_data = np.zeros((data.shape[0], data.shape[1], downsampled_length))
    print(downsampled_data.shape)
    downsampled_t_array = np.zeros(downsampled_length)
    for epoch, chan in product(range(data.shape[0]), range(data.shape[1])):
        convolved_data = np.convolve(
                                data[epoch, chan, :],
                                np.ones(window_size) / window_size, 
                                mode='valid'
                        )
        downsampled_data[epoch, chan, :] = convolved_data[:downsampled_length*window_size:window_size]
    downsampled_t_array = t_array[window_size//2 : downsampled_length * window_size : window_size]
    return downsampled_data, downsampled_t_array


def SvM_with_ECOC(data, labels, n_iter=1000):
    from sklearn.multiclass import OutputCodeClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from mne.decoding import Vectorizer, GeneralizingEstimator
    
    rs_val = 42
    clf_temporal = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        OutputCodeClassifier(
            estimator=SVC(kernel='rbf', probability=True),
            # estimator=SVC(kernel='linear', probability=True),
            code_size=2,
            random_state=rs_val
        )
    )
    gen_decod = GeneralizingEstimator(clf_temporal, n_jobs=-1, scoring='accuracy', verbose=True)
    # rs_val = 42
    # clf = OutputCodeClassifier(
    #     estimator=SVC(kernel='rbf', probability=True),
    #     code_size=5,
    #     random_state=rs_val
    # )

    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs_val)
    scores = cross_val_score(gen_decod, data, labels, cv=5, n_jobs=-1)
    # print(scores)
    print(np.mean(scores))
    return scores