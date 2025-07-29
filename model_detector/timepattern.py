import numpy as np
from numba import njit
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

@njit(fastmath=True)
def _z_score(subsequence):
    # Compute Z-scores for subsequence
    mean = np.mean(subsequence)
    std = np.std(subsequence)
    if std < 1e-6:
        return np.zeros_like(subsequence)
    return (subsequence - mean) / std

@njit(fastmath=True)
def _detect_outliers(subsequence, z_thresh, diff_multiplier):
    # Detect outliers using Z-score and difference thresholds
    z_scores = _z_score(subsequence)
    outliers = np.abs(z_scores) > z_thresh
    max_diff = np.max(np.abs(np.diff(subsequence)))
    if max_diff > diff_multiplier * np.std(subsequence):
        return True
    return np.any(outliers)

@njit(fastmath=True)
def _detect_trend(subsequence, trend_threshold):
    # Detect linear trend using least squares slope
    n = len(subsequence)
    if n < 2:
        return 0.0
    x_sum = 0.0
    y_sum = 0.0
    for i in range(n):
        x_sum += i
        y_sum += subsequence[i]
    x_mean = x_sum / n
    y_mean = y_sum / n
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        x_diff = i - x_mean
        y_diff = subsequence[i] - y_mean
        numerator += x_diff * y_diff
        denominator += x_diff * x_diff
    if abs(denominator) < 1e-6:
        return 0.0
    slope = numerator / denominator
    return slope

@njit(fastmath=True)
def _detect_volatility(subsequence, vol_inc_thresh, vol_dec_thresh):
    # Detect volatility changes between first/last quartiles
    quarter = len(subsequence) // 4
    if quarter < 2:
        return 0
    first_quarter = subsequence[:quarter]
    last_quarter = subsequence[-quarter:]
    std_first = np.std(first_quarter)
    std_last = np.std(last_quarter)
    if std_first < 1e-6 or std_last < 1e-6:
        return 0
    return std_last / std_first

def _detect_seasonality(subsequence, fixed_season_thresh, shift_season_thresh, period):
    # Detect seasonality patterns using decomposition
    if len(subsequence) < 2 * period:
        return False, False
    try:
        decomposition = seasonal_decompose(subsequence, period=period, extrapolate_trend='freq')
        seasonal = decomposition.seasonal
        seasonal_std = np.std(seasonal)
        fixed_seasonality = seasonal_std > fixed_season_thresh * np.std(subsequence)
        seasonal_changes = np.abs(np.diff(seasonal[::period]))
        shifting_seasonality = np.any(seasonal_changes > shift_season_thresh * np.std(subsequence))
        return fixed_seasonality, shifting_seasonality
    except Exception as e:
        return False, False

def detect_patterns(subsequence, period=12, thresholds=None):
    # Detect multiple patterns in time series subsequence
    if thresholds is None:
        thresholds = {'z_thresh': 2.5,'diff_multiplier': 3.0,'trend_threshold': 0.005,'fixed_season_thresh': 0.05,'shift_season_thresh': 0.3,'vol_inc_thresh': 1.3,'vol_dec_thresh': 0.77}
    patterns = []
    if _detect_outliers(subsequence, thresholds['z_thresh'], thresholds['diff_multiplier']):
        patterns.append("Outlier")
    trend_slope = _detect_trend(subsequence, thresholds['trend_threshold'])
    if trend_slope > thresholds['trend_threshold']:
        patterns.append("Upward Trend")
    elif trend_slope < -thresholds['trend_threshold']:
        patterns.append("Downward Trend")
    fixed_seasonal, shifting_seasonal = _detect_seasonality(subsequence, thresholds['fixed_season_thresh'], thresholds['shift_season_thresh'], period)
    if fixed_seasonal:
        patterns.append("Fixed Seasonality")
    if shifting_seasonal:
        patterns.append("Shifting Seasonality")
    volatility_ratio = _detect_volatility(subsequence, thresholds['vol_inc_thresh'], thresholds['vol_dec_thresh'])
    if volatility_ratio > thresholds['vol_inc_thresh']:
        patterns.append("Increased Volatility")
    elif volatility_ratio < thresholds['vol_dec_thresh']:
        patterns.append("Decreased Volatility")
    return patterns

def classify_subsequence(subsequence, period=12, thresholds=None):
    # Classify subsequence using priority-ordered pattern detection
    patterns = detect_patterns(subsequence, period, thresholds)
    if not patterns:
        return "Normal"
    priority_order = ["Outlier","Upward Trend", "Downward Trend","Fixed Seasonality", "Shifting Seasonality","Increased Volatility", "Decreased Volatility"]
    for pattern_type in priority_order:
        if pattern_type in patterns:
            return pattern_type
    return patterns[0]

def analyze_representative_windows(data, representative_windows, period=12, thresholds=None):
    # Analyze all representative windows in time series
    window_classifications = []
    detailed_patterns = []
    for start, end in representative_windows:
        subsequence = data[start:end]
        classification = classify_subsequence(subsequence, period, thresholds)
        patterns = detect_patterns(subsequence, period, thresholds)
        window_classifications.append(classification)
        detailed_patterns.append(patterns)
    return window_classifications, detailed_patterns