import numpy as np
import matplotlib.pyplot as plt
import torch
from model.net import Muti_UNet
from proceed import read_dat_file, notch_filter, highpass_filter, wavelet_denoise, resample_data, detect_qrs, \
    qualitycheck, calculate_representative_waveform, find_boundary, cal_T_on, weightcalculate

# file_path = '.\Data\HLT112\\20191119084500am-wrs.dat'
file_path = '.\Data\\xd0683.dat'
# file_path = '.\Data\\xd0692.dat'
# file_path = '.\Data\\xd0695.dat'
# 读取数据
ecg_data = read_dat_file(file_path)
# ecg_data = ecg_data[24000:, :]

data_hz = 200
target_hz = 500
# 去除基线漂移
resampled_ecg = highpass_filter(ecg_data, data_hz)
# 去除工频干扰
resampled_ecg = notch_filter(resampled_ecg, data_hz)
# 去除肌电干扰
resampled_ecg = wavelet_denoise(resampled_ecg)
# 重采样
resampled_ecg = resample_data(resampled_ecg, data_hz, target_hz)

channel_names = ['i', 'ii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
qrs_indices_per_channel = []
# 检查数据
time = np.arange(resampled_ecg.shape[0]) / target_hz # 生成时间轴，假设采样率为500Hz
representative_waveforms = []
for i in range(len(channel_names)):
    qrs_indices = detect_qrs(resampled_ecg[:, 1], target_hz)
    overlays = qualitycheck(resampled_ecg[:, i], qrs_indices, target_hz)
    qrs_indices_per_channel.append(qrs_indices)

    extra_zeros = np.zeros(12)
    representative_waveform = calculate_representative_waveform(overlays)
    representative_waveform = np.concatenate((representative_waveform, extra_zeros))
    representative_waveforms.append(representative_waveform)

model = Muti_UNet()
path = '.\modelRecord\\LUDB_0.001_0.01_best_loc.pt'
checkpoint = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()
loc = {'p_pre': [], 'qrs_pre': [], 't_pre': [], 'fs': []}
weighted_loc = {'p_on': 0, 'p_end': 0, 'qrs_on': 0, 'qrs_end': 0, 't_peak': 0, 't_end': 0}
location = {'p_on': [], 'p_end': [], 'qrs_on': [], 'qrs_end': [], 't_peak': [], 't_end': []}
p_pre = []
qrs_pre = []
t_pre = []
with (torch.no_grad()):
    for i, waveform in enumerate(representative_waveforms):
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        waveform_tensor.reshape(1, 512)
        locations = model(waveform_tensor)

        p = locations[:, 0, :].detach().numpy().squeeze()
        qrs = locations[:, 1, :].detach().numpy().squeeze()
        t = locations[:, 2, :].detach().numpy().squeeze()

        loc['p_pre'].append(p)
        loc['qrs_pre'].append(qrs)
        loc['t_pre'].append(t)

        p_on, p_end = find_boundary(p, 0.6)
        qrs_on, qrs_end = find_boundary(qrs, 0.6)
        t_peak, t_end = find_boundary(t, 0.5)

        location['p_on'].append(p_on)
        location['p_end'].append(p_end)
        location['qrs_on'].append(qrs_on)
        location['qrs_end'].append(qrs_end)
        location['t_peak'].append(t_peak)
        location['t_end'].append(t_end)

weight = weightcalculate(representative_waveforms,location,0.8,0.5,0.8)
for i, waveform in enumerate(representative_waveforms):
    weighted_loc['p_on'] += location['p_on'][i] * weight['p'][i]
    weighted_loc['p_end'] += location['p_end'][i] * weight['p'][i]
    weighted_loc['qrs_on'] += location['qrs_on'][i] * weight['qrs'][i]
    weighted_loc['qrs_end'] += location['qrs_end'][i] * weight['qrs'][i]
    weighted_loc['t_peak'] += location['t_peak'][i] * weight['t'][i]
    weighted_loc['t_end'] += location['t_end'][i] * weight['t'][i]

t_on = cal_T_on(location['t_peak'], representative_waveforms)
weighted_loc['t_on'] = t_on
print("final p_on:", weighted_loc['p_on'])
print("final p_end:", weighted_loc['p_end'])
print("final qrs_on:", weighted_loc['qrs_on'])
print("final qrs_end:", weighted_loc['qrs_end'])
print("final t_on:", weighted_loc['t_on'])
print("final t_peak:", weighted_loc['t_peak'])
print("final t_end:", weighted_loc['t_end'])

plt.figure(figsize=(18, 18))
# 同时显示八个通道的代表波形
for i, name in enumerate(channel_names):
    plt.subplot(5, 2, i + 1)
    plt.plot(representative_waveforms[i], color='black')
    # plt.plot(loc['p_pre'][i], color='orange')
    # plt.plot(loc['qrs_pre'][i], color='blue')
    # plt.plot(loc['t_pre'][i], color='red')
    plt.title(f'{name} Representative Waveform')
    plt.xlabel('Samples (500Hz)')
    plt.ylabel('mV')
# 叠加显示所有通道的代表波形

plt.subplot(5, 2, 9)
for i, name in enumerate(channel_names):
    plt.plot(representative_waveforms[i],color = 'black')
    plt.axvline(weighted_loc['p_on'], linestyle='-', alpha=0.5, color = 'orange')
    plt.axvline(weighted_loc['p_end'], linestyle='-', alpha=0.5,color = 'orange')
    plt.axvline(weighted_loc['qrs_on'], linestyle='-', alpha=0.5,color = 'green')
    plt.axvline(weighted_loc['qrs_end'], linestyle='-', alpha=0.5,color = 'green')
    plt.axvline(weighted_loc['t_on'], linestyle='-', alpha=0.5, color='red')
    plt.axvline(weighted_loc['t_peak'], linestyle='-', alpha=0.5,color = 'red')
    plt.axvline(weighted_loc['t_end'], linestyle='-', alpha=0.5,color = 'red')

plt.title('Representative Waveforms for All Channels')
plt.xlabel('Samples (500Hz)')
plt.ylabel('mV')
plt.legend()
plt.tight_layout()
plt.show()