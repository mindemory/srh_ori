import numpy as np
from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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
from sklearn.multiclass import OutputCodeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score


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


def process_time_point(time_point_train, data, labels, unique_labels, n_subsets, n_iter, rs_val):
    n_time_points = data.shape[2] # Number of time points present in this data
    score_this_time_point = np.empty((n_time_points,n_iter))
    
    print(f"Running for time point {time_point_train}")
    
    
    # for time_point_train in range(n_time_points):
        
    # Run a model at each time point
    # PS: make sure that the data is downsampled before passing in
    for iteration in range(n_iter):
        averaged_data_train = []
        averaged_labels_train = []
        averaged_data_test = []
        averaged_labels_test = []

        for class_label in unique_labels:
            # Generate training data set at this time point
            class_data_train = data[labels == class_label, :, time_point_train]
            n_trials_train = class_data_train.shape[0]
            subset_size_train = n_trials_train // n_subsets
            for subset in range(n_subsets):
                subset_trials_train = class_data_train[subset*subset_size_train : (subset+1)*subset_size_train]
                averaged_data_train.append(subset_trials_train.mean(axis=0))
                averaged_labels_train.append(class_label)

            # Generate test data set at all other time points
            class_data_test = data[labels == class_label, :, :]
            n_trials_test = class_data_test.shape[0]
            subset_size_test = n_trials_test // n_subsets
            for subset in range(n_subsets):
                subset_trials_test = class_data_test[subset*subset_size_test : (subset+1)*subset_size_test]
                averaged_data_test.append(subset_trials_test.mean(axis=0))
                averaged_labels_test.append(class_label)
        
        averaged_data_train = np.array(averaged_data_train)
        averaged_labels_train = np.array(averaged_labels_train)
        averaged_data_test = np.array(averaged_data_test)
        averaged_labels_test = np.array(averaged_labels_test)

        clf = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            OutputCodeClassifier(
                estimator=SVC(kernel='rbf', probability=True),
                code_size=2,
                random_state=rs_val+iteration
            )
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs_val+iteration)
        for train_idx, test_idx in cv.split(averaged_data_train, averaged_labels_train):
            X_train, X_test = averaged_data_train[train_idx], averaged_data_test[test_idx]
            y_train, y_test = averaged_labels_train[train_idx], averaged_labels_test[test_idx]

            clf.fit(X_train, y_train)
            for time_point_test in range(n_time_points):
                y_pred = clf.predict(X_test[:, :, time_point_test])
                score_this_test_point = accuracy_score(y_test, y_pred)
                score_this_time_point[time_point_test, iteration] = score_this_test_point
    return score_this_time_point


def SvM_with_ECOC(data, behav_df, method='trialwise'):
    rs_val = 42
    y_ori = behav_df['targOriCategory'].values
    y_hemi = behav_df['stimPF'].values
    if method == 'trialwise':
        labels = y_hemi
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
        gen_decod = GeneralizingEstimator(clf_temporal, n_jobs=-1, scoring=None, verbose=True)
        # rs_val = 42
        # clf = OutputCodeClassifier(
        #     estimator=SVC(kernel='rbf', probability=True),
        #     code_size=5,
        #     random_state=rs_val
        # )
        # scores_matrix = estimator.score(data, labels)
        def average_score(estimator, X, y):
            scores_matrix = estimator.score(X, y)
            # return np.mean(scores_matrix)
            return scores_matrix

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs_val)
        scores = cross_val_score(gen_decod, data, labels, cv=cv, n_jobs=-1, scoring=average_score)
        print(scores.shape)
        print(np.mean(scores))
        return scores
    elif method == 'grouped_conds':
        # labels = [-1 * y_ori[i] if y_hemi[i] == 0 else y_ori[i] for i in range(len(y_ori))]
        labels = y_ori
        print(np.unique(labels))
        unique_labels = np.unique(labels)
        n_time_points = data.shape[2] # Number of time points present in this data
        n_iter = 3
        n_classes = len(unique_labels)
        
        n_subsets = 3
        print(f"We are running on {n_subsets} subsets with {data[labels==unique_labels[0]].shape[0]//n_subsets} trials each")
        
        all_scores = np.empty((n_time_points, n_iter, n_classes))
        for time_point in range(n_time_points):
            # Run a model at each time point
            # PS: make sure that the data is downsampled before passing in
            for iteration in range(n_iter):
                averaged_data = []
                averaged_labels = []

                for class_label in unique_labels:
                    class_data = data[labels == class_label, :, time_point]
                    n_trials = class_data.shape[0]
                    subset_size = n_trials // n_subsets
                    for subset in range(n_subsets):
                        subset_trials = class_data[subset*subset_size : (subset+1)*subset_size]
                        averaged_data.append(subset_trials.mean(axis=0))
                        averaged_labels.append(class_label)
                averaged_data = np.array(averaged_data)
                averaged_labels = np.array(averaged_labels)

                clf = make_pipeline(
                    Vectorizer(),
                    StandardScaler(),
                    OutputCodeClassifier(
                        estimator=SVC(kernel='rbf', probability=True),
                        code_size=2,
                        random_state=rs_val+iteration
                    )
                )

                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs_val+iteration)
                for train_idx, test_idx in cv.split(averaged_data, averaged_labels):
                    X_train, X_test = averaged_data[train_idx], averaged_data[test_idx]
                    y_train, y_test = averaged_labels[train_idx], averaged_labels[test_idx]
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    score_this_time_point = accuracy_score(y_test, y_pred)
                    all_scores[time_point, iteration] = score_this_time_point
                    
        avg_scores = np.mean(all_scores, axis=(1, 2))
        smoothed_scores = np.convolve(avg_scores, np.ones(5)/5, mode='same')
        return smoothed_scores, 1/n_classes

    elif method == 'TGA_grouped_conds':
        import multiprocessing as mp
        labels = [-1 * y_ori[i] if y_hemi[i] == 0 else y_ori[i] for i in range(len(y_ori))]
        print(np.unique(labels))
        unique_labels = np.unique(labels)
        n_time_points = data.shape[2] # Number of time points present in this data
        n_iter = 3
        n_classes = len(unique_labels)
        
        n_subsets = 3
        print(f"We are running on {n_subsets} subsets with {data[labels==unique_labels[0]].shape[0]//n_subsets} trials each")
        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(
            process_time_point,
            [(time_point, data, labels, unique_labels, n_subsets, n_iter, rs_val) for time_point in range(n_time_points)]
        )
        pool.close()
        pool.join()
        
        all_scores = np.empty((n_time_points, n_time_points, n_iter))
        for idx, result in enumerate(results):
            all_scores[idx, :, :] = result
        
        return all_scores, 1/n_classes





                    # X_train, X_test = averaged_data[train_idx], averaged_data[test_idx]
                    # y_train, y_test = averaged_labels[train_idx], averaged_labels[test_idx]
                    # clf.fit(X_train, y_train)
                    # y_pred = clf.predict(X_test)
                    # score_this_time_point = accuracy_score(y_test, y_pred)
                    # all_scores[time_point, iteration] = score_this_time_point
                    
        # avg_scores = np.mean(all_scores, axis=(1, 2))
        # smoothed_scores = np.convolve(avg_scores, np.ones(5)/5, mode='same')
        # return smoothed_scores, 1/n_classes

