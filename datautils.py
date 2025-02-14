import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_ftn(cv_num, seg_length=1280, is_train=True, overlap_test=128):
    """
    Load and segment velocity data for the finger-to-nose task, returning a cross-validation dataset.

    Parameters:
    - cv_num: Number of cross-validation folds.
    - seg_length: Length of each segment (in samples).
    - is_train: Boolean indicating whether the data is for training (True) or testing (False).
    - overlap_test: Overlap between segments during testing (in samples).

    Returns:
    - data_set: Dictionary containing segmented and standardized data with keys for each fold.
      Each fold contains:
        - 'data': Segmented and standardized velocity time-series data.
        - 'total_bars': Total BARS scores for each segment.
        - 'arm_bars': Combined arm BARS finger-to-nose subscores for each segment.
        - 'larm_bars': Left arm BARS finger-to-nose subscores for each segment.
        - 'rarm_bars': Right arm BARS finger-to-nose subscores for each segment.
        - 'sub_date': Subject and date identifiers for each segment.
        - 'arm': Arm (L or R) for each segment.
        - 'diag': Diagnosis (1 for ataxia, 0 for healthy) for each segment.
        - 'age': Age of the subject for each segment.
        - 'hand': Dominant hand of the subject for each segment.
        - 'sex': Sex of the subject for each segment.
        - 'timestamp': Start and end indices of each segment.
    """
    # Load velocity time-series
    data = np.load('./FNT_data/DecomposedMovements_norotation.npz', allow_pickle=True)
    vel_all = data['full_vel'][()]

    # Load clinician-scored severity and demographic information
    labels = pd.read_csv('./FNT_data/FNF_labels.csv')

    diagnoses_dict = {
        (int(row['ID'][:5]), row['ID'][6:]): row
        for _, row in labels.iterrows()
    }
    # for training set: segment time-series using sliding window with an overlap of $L/2$,
    # for testing set: with no-overlapping
    seg_offset = seg_length // 2 if is_train else overlap_test

    vel_seg, meta_seg = [], []

    for (s, dt), diag_info in diagnoses_dict.items():
        diag = diag_info['gen_diagnosis_num']
        # include only ataxia and healthy
        # 1 = ataxia, 7 = pediatric with ataxia, 3 = healthy
        if s == 10234 or diag not in {1, 3, 7}:  # excluding a subject whose data is not available
            continue

        for a in ('L', 'R'):  # left and right arm
            key = (s, dt, a)  # subject id, date, arm
            if key not in vel_all:
                continue

            vel = vel_all[key]
            num_segments = (len(vel) - seg_length) // seg_offset + 1

            # segment velocity time-series into input segments
            for k in range(num_segments):
                start = k * seg_offset
                end = start + seg_length
                vel_seg.append(vel[start:end, :])

                bars_total = diag_info['bars_total'] if diag != 3 else 0
                bars_arm_L = diag_info['bars_arm_L'] if diag != 3 else 0
                bars_arm_R = diag_info['bars_arm_R'] if diag != 3 else 0

                meta_seg.append((s, f"{s}_{dt}", a, bars_total, bars_arm_L + bars_arm_R,
                                 bars_arm_L, bars_arm_R, diag_info['age'], diag_info['hand'],
                                 diag_info['sex'], 1 if diag in {1, 7} else 0, (start, end)))


    vel_seg = np.array(vel_seg)
    meta_seg = np.array(meta_seg, dtype=object)

    # shuffle subjects
    unique_subjects = np.unique(meta_seg[:, 0])
    np.random.shuffle(unique_subjects)

    # split the dataset into 5 folds
    data_set = {}
    num_per_cv = len(unique_subjects) // cv_num
    for fold in range(cv_num):
        if fold == (cv_num - 1):
            test_sub = unique_subjects[fold * num_per_cv:]
        else:
            test_sub = unique_subjects[fold * num_per_cv:(fold + 1) * num_per_cv]

        train_sub = unique_subjects[np.isin(unique_subjects, test_sub, invert=True)]
        target_sub = train_sub if is_train else test_sub

        mask_train = np.isin(meta_seg[:, 0], train_sub)
        mask_target = np.isin(meta_seg[:, 0], target_sub)

        scaler = StandardScaler()
        scaler.fit(vel_seg[mask_train].reshape(-1, vel_seg.shape[-1]))
        target_data = scaler.transform(vel_seg[mask_target].reshape(-1, vel_seg.shape[-1]))

        data_set[fold] = {
            'data':target_data.reshape(-1, seg_length, vel_seg.shape[-1]),
            'total_bars': meta_seg[mask_target, 3].astype(float),
            'arm_bars': meta_seg[mask_target, 4].astype(float),
            'larm_bars': meta_seg[mask_target, 5].astype(float),
            'rarm_bars': meta_seg[mask_target, 6].astype(float),
            'sub_date': meta_seg[mask_target, 1],
            'arm': meta_seg[mask_target, 2],
            'diag': meta_seg[mask_target, 10].astype(int),
            'age': meta_seg[mask_target, 7].astype(float),
            'hand': meta_seg[mask_target, 8],
            'sex': meta_seg[mask_target, 9],
            'timestamp': meta_seg[mask_target, 11]
        }
                       
    return data_set
