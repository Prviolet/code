from scipy.signal import find_peaks, resample, butter, filtfilt, iirnotch
import numpy as np
from scipy.stats import norm
import pywt
def normcdf(x):
    return norm.cdf(x)
def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))
def mean_minmax(ecg_data):
    # 将输入数据转换为NumPy数组
    min_val = ecg_data.min()
    max_val = ecg_data.max()
    normalized_data = 2 * (ecg_data - min_val) / (max_val - min_val) - 1
    return normalized_data.tolist()
def read_dat_file(file_path, duration_minutes=6, sample_rate=200, sensitivity=900):
    sample_bytes = 2
    frame_bytes = 16
    channels = 8
    duration_seconds = duration_minutes * 60
    total_samples = duration_seconds * sample_rate
    with open(file_path, 'rb') as f:
        data = f.read(total_samples * frame_bytes)
    data = np.frombuffer(data, dtype=np.int16)
    data = data.reshape(-1, channels)
    data = data / sensitivity
    return data
def resample_data(data, original_rate, target_rate):
    num_samples = int(data.shape[0] * target_rate / original_rate)
    resampled_data = resample(data, num_samples, axis=0)
    return resampled_data
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def highpass_filter(data, fs,):
    b, a = butter_highpass(0.5, fs, 5)
    y = filtfilt(b, a, data, axis=0)
    return y
def notch_filter(data,fs):
    nyquist = 0.5 * fs
    normal_cutoff = 50.0 / nyquist
    b, a = iirnotch(normal_cutoff, 30)
    y = filtfilt(b, a, data, axis=0)
    return y
def wavelet_denoise(data, wavelet='bior4.4', level=5):
    coeffs = pywt.wavedec(data, wavelet, level=level, axis=0)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    denoised_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    denoised_data = pywt.waverec(denoised_coeffs, wavelet, axis=0)
    return denoised_data
def detect_qrs(ecg_channel, sample_rate):
    distance = int(0.5 * sample_rate)  # Adjust distance for lower sample rate
    height_threshold = np.mean(ecg_channel) + 0.5 * np.std(ecg_channel)
    peaks, _ = find_peaks(ecg_channel, distance=distance, height=height_threshold)
    return peaks
def generate_overlay_plots(ecg_lead, qrs_indices, window_size):
    overlays = []
    half_window = window_size // 2
    for idx in qrs_indices:
        # 检查窗口是否超出ECG信号的范围
        if idx - half_window < 0 or idx + half_window > len(ecg_lead):
            continue
        # 提取窗口内的ECG波形
        overlay = ecg_lead[idx - half_window: idx + half_window]
        overlays.append(overlay)
    return np.array(overlays)
def calculate_representative_waveform(overlays):
    return np.mean(overlays, axis=0)
def qualitycheck(ecg_lead, R_peak, frequency):
    cuttings = []
    half_window = frequency // 2
    for idx in R_peak:
        if idx - half_window < 0 or idx + half_window > len(ecg_lead):
            continue
        cutting = ecg_lead[idx - half_window: idx + half_window]
        cuttings.append(cutting)            # 分割出心拍片段

    quality_factor = [1] * len(cuttings)    #输出的质量因数
    beat_len = len(cuttings[0])
    normal_ecg_data = [mean_minmax(data) for data in cuttings]  # 心拍数据归一化
    pass_quality = [1] * len(cuttings)
    for i, normal_ecg_data_item in enumerate(normal_ecg_data):
        min_val = min(normal_ecg_data_item)
        max_val = max(normal_ecg_data_item)
        negative_count_1 = sum(
            (normal_ecg_data_item[j] - 0.75 * max_val) * (normal_ecg_data_item[j + 1] - 0.75 * max_val) < 0      #25分位线
            for j in range(len(normal_ecg_data_item) - 1)
        )
        negative_count_2 = sum(
            (normal_ecg_data_item[j] - max_val / 2) * (normal_ecg_data_item[j + 1] - max_val / 2) < 0            #50分位线
            for j in range(len(normal_ecg_data_item) - 1)
        )
        negative_count_3 = sum(
            (normal_ecg_data_item[j] - max_val / 4) * (normal_ecg_data_item[j + 1] - max_val / 4) < 0            #75分位线
            for j in range(len(normal_ecg_data_item) - 1)
        )

        negative_count_4 = sum(
            (normal_ecg_data_item[j] - 0.75 * min_val) * (normal_ecg_data_item[j + 1] - 0.75 * min_val) < 0      #25分位线
            for j in range(len(normal_ecg_data_item) - 1)
        )
        negative_count_5 = sum(
            (normal_ecg_data_item[j] - min_val / 2) * (normal_ecg_data_item[j + 1] - min_val / 2) < 0            #50分位线
            for j in range(len(normal_ecg_data_item) - 1)
        )
        negative_count_6 = sum(
            (normal_ecg_data_item[j] - min_val / 4) * (normal_ecg_data_item[j + 1] - min_val / 4) < 0            #75分位线
            for j in range(len(normal_ecg_data_item) - 1)
        )
        if negative_count_1 <= 1 and negative_count_2 <= 1 and negative_count_3 <= 1:
            pass_quality[i] = 0
        # if negative_count_3 > 3 or negative_count_6 > 2:


    RR_diff_list = [
        (R_peak[i] - R_peak[i - 1]) if i > 0 else 0
        for i in range(len(cuttings))
    ]
    miu_p1 = 0.89 * frequency
    sigma_p1 = 0.28 * frequency
    RR_diff_z1 = [(RR - miu_p1) / sigma_p1 for RR in RR_diff_list]      #计算所有RR_diff因子
    p_value1 = [normcdf(z) for z in RR_diff_z1]
    RR_diff_p1 = [1 - abs(p - 0.5) / 0.5 for p in p_value1]
    pdist2_num = 1
    beat_pdist2_list = [
        [
            euclidean_distance(normal_ecg_data[i], normal_ecg_data[i + j + 1]) if j < pdist2_num - (i + 1) + 1 else 0
            for j in range(2 * pdist2_num + 1)
        ]
        for i in range(len(cuttings))
    ]
    max_pdist2_distance = beat_len
    for i in range(len(beat_pdist2_list)):
        for j in range(len(beat_pdist2_list[0])):
            beat_pdist2_list[i][j] = 1 - beat_pdist2_list[i][j] / max_pdist2_distance
    pdist2_average = [np.mean(beat_pdist2_list[i]) for i in range(len(beat_pdist2_list))]
    range_list = [max(data) - min(data) for data in cuttings]
    count_num = 1
    range_quality = []
    for i in range(len(range_list)):
        start_ind = max(0, i - count_num)
        end_ind = min(len(range_list) - 1, i + count_num)
        range_sum = sum(range_list[start_ind:end_ind + 1])
        temp_denominator = range_sum - range_list[i]
        if temp_denominator == 0:
            temp_denominator = 0.001      #避免除零
        range_quality.append(range_list[i] / temp_denominator)
    for i in range(len(cuttings)):
        if pass_quality[i] < 0.05 or pdist2_average[i] < 0.976 or RR_diff_p1[i] < 0.05 or range_quality[i] > 10:
            quality_factor[i] = 0

    cuttings = [cuttings[j] for j in range(len(cuttings)) if quality_factor[j] == 1]
    return np.array(cuttings)
def weightcalculate(ecg_leads,location,alpha_p,alpha_qrs,alpha_t):
    weight = {'p': [], 'qrs': [], 't': []}
    sum_p = 0
    sum_qrs = 0
    sum_t = 0
    for i in range(len(ecg_leads)):
        if location['p_on'][i] == 0 or location['p_end'][i] == 0:
            continue
        else:
            sum_p += np.exp(alpha_p * abs(ecg_leads[i][int(location['p_on'][i]):int(location['p_end'][i])]).mean())
        if location['qrs_on'][i] == 0 or location['qrs_end'][i] == 0:
            continue
        else:
            sum_qrs += np.exp(alpha_qrs * abs(ecg_leads[i][int(location['qrs_on'][i]):int(location['qrs_end'][i])]).mean())
        if location['t_peak'][i] == 0 or location['t_end'][i] == 0:
            continue
        else:
            sum_t += np.exp(alpha_t * abs(ecg_leads[i][int(location['t_peak'][i])]).mean())

    for i in range(len(ecg_leads)):
        if location['p_on'][i] == 0 or location['p_end'][i] == 0:
            weight['p'].append(0)
        else:
            weight['p'].append(np.exp(alpha_p * abs(ecg_leads[i][int(location['p_on'][i]):int(location['p_end'][i])]).mean())/sum_p)
        if location['qrs_on'][i] == 0 or location['qrs_end'][i] == 0:
            weight['qrs'].append(0)
        else:
            weight['qrs'].append(np.exp(alpha_qrs * abs(ecg_leads[i][int(location['qrs_on'][i]):int(location['qrs_end'][i])]).mean())/sum_qrs)
        if location['t_peak'][i] == 0 or location['t_end'][i] == 0:
            weight['t'].append(0)
        else:
            weight['t'].append(np.exp(alpha_t * abs(ecg_leads[i][int(location['t_peak'][i])]).mean())/sum_t)

    return weight
def find_boundary(data, thr):
    data = data.reshape(-1)
    # 找到波形大于thr的位置
    R_index = np.where(data > thr)[0]
    if len(R_index) == 0:
        return 0, 0
    # 前后位置相减作差，如果预测波形不止一个方波，则 R_diff 中会出现大于1的值
    R_diff = np.diff(R_index).reshape(-1)
    # 这里的 30 是一个阈值，因为不止一个方波则 R_diff 中肯定会出现大于1的值，观察R_diff并且为防止干扰，这个值设大一点
    R_end = np.where(R_diff > 30)[0]
    if len(R_end) == 0:
        R_end = np.array([R_index[-1]])
    else:
        R_end = R_index[R_end]
        R_end = np.concatenate((R_end, np.array([R_index[-1]])), axis=0)
    # 起点
    # 找到波形大于thr的位置
    R_index_ = R_index[::-1]
    # 前后位置相减作差，如果预测波形不止一个方波，则 R_diff_ 中会出现大于1的值
    R_diff_ = np.diff(R_index_).reshape(-1)
    R_on = np.where(R_diff_ < -30)[0]
    if len(R_on) == 0:
        R_on = np.array([R_index_[-1]])
    else:
        R_on = R_index_[R_on]
        R_on = np.concatenate((R_on, np.array([R_index_[-1]])), axis=0)
        R_on = R_on[::-1]
    return R_on, R_end
def cal_T_on(location, leads):
    t_peak_max = []
    leads = np.array(leads)
    for i in range(leads.shape[0]):
        t_peak_value = leads[i][location[i]]
        t_peak_max.append((t_peak_value, i))

    t_peak_max.sort(reverse=True, key=lambda x: x[0])
    top_two_channels = [t_peak_max[0][1], t_peak_max[1][1]]

    t_start_points = []
    for channel in top_two_channels:
        t_peak = int(location[channel])
        for i in range(t_peak, 0, -1):
            if leads[channel][i] * leads[channel][i - 1] <= 0:
                t_start = i
                break
        t_start_points.append(t_start)
    return np.mean(t_start_points)