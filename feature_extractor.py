import numpy as np
def extract_features(audio: np.ndarray, sr: int, frame_length: int, 
                     zcr_threshold: float = 0.15, vol_threshold: float = 0.1):
    """ Extracts frame-level features from the given audio signal. """

    step = frame_length
    frames = [audio[i:i + frame_length] for i in range(0, len(audio) - frame_length + 1, step)]

    # min and max volumes for better thresholding
    volumes = [np.sqrt(np.mean(frame ** 2)) for frame in frames]
    min_vol = np.min(volumes)
    max_vol = np.max(volumes)

    features = []
    for frame in frames:
        ste = np.mean(frame ** 2)
        vol = np.sqrt(ste)
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
        f0 = estimate_f0(frame, sr)
        
        silent_ratio = int(vol < min_vol + (max_vol - min_vol) * vol_threshold and zcr > 0.001)  
        voiced_ratio = int(zcr < zcr_threshold and vol > min_vol + (max_vol - min_vol) * vol_threshold and f0 < 1000)
        
        features.append((ste, vol, zcr, silent_ratio, f0, voiced_ratio))

    return features

def extract_clip_features(features):
    """ Extracts clip-level features from the given frame-level features. """
    clip_features = {}

    # volume-based features
    volume = features['volume']
    clip_features['VSTD'] = np.std(volume) / np.max(volume) if np.max(volume) > 0 else 0
    clip_features['VDR'] = (np.max(volume) - np.min(volume)) / np.max(volume) if np.max(volume) > 0 else 0
    clip_features['VU'] = np.sum(np.abs(np.diff(volume)))  # Sum of differences between peaks & valleys

    # energy-based features
    ste = features['STE']
    avSTE = np.mean(ste)
    clip_features['LSTER'] = np.sum((ste < 0.5 * avSTE).astype(int)) / len(ste)

    energy_segments = np.array_split(ste, 10)
    segment_energies = [np.sum(segment) for segment in energy_segments]
    normalized_energies = segment_energies / np.sum(segment_energies)
    clip_features['Energy_Entropy'] = -np.sum(normalized_energies * np.log2(normalized_energies + 1e-10))

    # ZCR-based features
    zcr = features['ZCR']
    clip_features['ZSTD'] = np.std(zcr)
    avZCR = np.mean(zcr)
    clip_features['HZCRR'] = np.sum((zcr > 1.5 * avZCR).astype(int)) / len(zcr)

    return clip_features

def extract_mini_clip_features(features, sr, frame_length=441, clip_length=1.0, step=0.5):
    """ Extracts clip-level features for overlapping 1-second clips with a step of 0.5 seconds. """
    clip_length_frames = int(clip_length * sr / frame_length)
    step_length_frames = int(step * sr / frame_length)
    mini_clip_features = []
    num_clips = max(1, (len(features) - clip_length_frames) // step_length_frames + 1)+1

    for i in range(num_clips):
        clip_start = i * step_length_frames
        clip_end = min(clip_start + clip_length_frames, len(features))
        segment_features = features[clip_start:clip_end]
        
        if len(segment_features) == 0:
            continue
        
        features_dict = {
            'STE': [f[0] for f in segment_features],  
            'volume': [f[1] for f in segment_features],  
            'ZCR': [f[2] for f in segment_features], 
        }
        mini_clip_features.append(extract_clip_features(features_dict))

    return mini_clip_features

def amdf(frame, min_lag=40, max_lag=320):
    """Compute AMDF and return the best lag within a valid range."""
    frame_length = len(frame)
    amdf_values = np.zeros(max_lag)

    for lag in range(min_lag, max_lag):  
        amdf_values[lag] = np.sum(np.abs(frame[:frame_length - lag] - frame[lag:]))
    best_lag = np.argmin(amdf_values[min_lag:max_lag]) + min_lag
    return best_lag

def autocorrelation_manual(frame):
    """Computes the autocorrelation function manually and finds the first peak."""
    N = len(frame)
    autocorr = np.zeros(N)
    for lag in range(N):
        autocorr[lag] = np.sum(frame[:N-lag] * frame[lag:N])

    autocorr /= np.max(autocorr)
    peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[0] + 1
    
    return peaks[0] if len(peaks) > 0 else 0

def estimate_f0(frame, sr, type='other'):
    """
    Estimates the fundamental frequency (F0) of a single frame.
    
    Parameters:
    - frame: numpy array (audio frame)
    - sr: sample rate

    Returns:
    - f0: estimated fundamental frequency
    """
    min_lag_amdf = amdf(frame)
    f0_amdf = sr / min_lag_amdf if min_lag_amdf > 0 else 0

    peak_lag_autocorr = autocorrelation_manual(frame)
    f0_autocorr = sr / peak_lag_autocorr if peak_lag_autocorr > 0 else 0

    if type == 'autocorr':
        return f0_autocorr
    else: 
        return f0_amdf

def format_time(ms):
    """Formats milliseconds to HH:MM:SS.MMM format."""
    s = ms // 1000 
    m, s = divmod(s, 60) 
    h, m = divmod(m, 60)  
    ms_remaining = ms % 1000
    return f"{h}:{m:02}:{s:02}.{ms_remaining:03}"

def custom_correlate(x, y):
    """
    Compute the cross-correlation of two 1D arrays.
    Assumes both input arrays are of the same length.
    """
    N = len(x)
    result = [0] * N
    
    for lag in range(N):
        sum_val = 0
        for i in range(N - lag):
            sum_val += x[i] * y[i + lag]
        result[lag] = sum_val
    
    return result