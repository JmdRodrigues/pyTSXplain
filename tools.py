import numpy as np
import numpy.fft as fft
from scipy import stats
import pycatch22 as catch22

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def mov_sum_mean_std(ts,m):
    """
    Calculate the standard deviation within a moving window.
    Parameters
    ----------
    ts: signal.
    m: moving window size.
    """
    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")
    #Add zero to the beginning of the cumsum of ts
    s = np.insert(np.cumsum(ts),0,0)
    #Add zero to the beginning of the cumsum of ts ** 2
    sSq = np.insert(np.cumsum(ts ** 2),0,0)
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] -sSq[:-m]

    return segSum, segSum/m, np.sqrt(segSumSq / m - (segSum/m) ** 2)

def slidingDotProduct(query, ts):
    """
    Calculate the dot product between a query and all subsequences of length(query) in the timeseries ts. Note that we use Numpy's rfft method instead of fft.
    Parameters
    ----------
    query: Specific time series query to evaluate.
    ts: Time series to calculate the query's sliding dot product against.
    """
    m = len(query)
    n = len(ts)

    #If length is odd, zero-pad time time series
    ts_add = 0
    if n%2 ==1:
        ts = np.insert(ts,0,0)
        ts_add = 1

    q_add = 0
    #If length is odd, zero-pad query
    if m%2 == 1:
        query = np.insert(query,0,0)
        q_add = 1

    #This reverses the array
    query = query[::-1]

    query = np.pad(query,(0,n-m+ts_add-q_add),'constant')

    #Determine trim length for dot product. Note that zero-padding of the query has no effect on array length, which is solely determined by the longest vector
    trim = m-1+ts_add

    dot_product = fft.irfft(fft.rfft(ts)*fft.rfft(query))

    #Note that we only care about the dot product results from index m-1 onwards, as the first few values aren't true dot products (due to the way the FFT works for dot products)
    return dot_product[trim :]


def mass(query, ts):
    """
    Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS): a Euclidian distance similarity search algorithm. Note that we are returning the square of MASS.
    Parameters
    ----------
    :query: Time series snippet to evaluate. Note that the query does not have to be a subset of ts.
    :ts: Time series to compare against query.
    """

    #query_normalized = zNormalize(np.copy(query))
    m = len(query)
    q_mean = np.mean(query)
    q_std = np.std(query)
    sum_, mean, std = mov_sum_mean_std(ts, m)
    dot = slidingDotProduct(query, ts)

    #res = np.sqrt(2*m*(1-(dot-m*mean*q_mean)/(m*std*q_std)))
    res = 2*m*(1-(dot-m*mean*q_mean)/(m*std*q_std))

    return res

def gauss_val_moving(sig, c, win_size):
    a = 1
    b = 0
    x = np.linspace(-1, 1, win_size)
    query_valley_ = -a * np.exp(((x - b) ** 2) / (-2 * (c ** 2)))
    mass_val = np.min(mass(query_valley_, sig))

    return mass_val

def gauss_peak_moving(sig, c, win_size):
    a = 1
    b = 0
    x = np.linspace(-1, 1, win_size)
    query_peak = a * np.exp(((x - b)**2) / (-2 * (c**2)))
    mass_peak = np.min(mass(query_peak, sig))

    return mass_peak

def slope(y):
    X = np.linspace(1, len(y), len(y))
    return np.polyfit(X, y, 1)[0]

def slope2angle(slope_val):
    return np.arctan(slope_val)*(180/np.pi)

def word_feature_vectors_extraction(X, norm=True):
    X_words = np.zeros((np.shape(X)[0], 5))
    for i in range(np.shape(X)[0]):
        sig_i = X[i, :]
        moving_sl = slope(sig_i)
        moving_up = slope2angle(moving_sl) if moving_sl > 0 else 0
        moving_down = slope2angle(moving_sl) if moving_sl < 0 else 0
        moving_flat = 90-abs(slope2angle(moving_sl))

        peak_ = gauss_peak_moving(sig_i, 0.6, len(sig_i))
        valley_ = gauss_val_moving(sig_i, 0.6, len(sig_i))

        words_ = [moving_up, moving_down, moving_flat, peak_, valley_]
        X_words[i, :] = words_

    if(norm):
        scaler = StandardScaler()
        X_words = scaler.fit_transform(X_words)

    return X_words, ["up", "down", "flat", "peak", "valley"]

def catch22_feature_extraction(X, norm=True):
    X_catch22 = np.zeros((np.shape(X)[0], 22))
    for i in range(np.shape(X)[0]):
        sig_i = X[i, :]
        catch22_i = catch22.catch22_all(sig_i)
        X_catch22[i, :] = catch22_i["values"]
    if(norm):
        scaler = StandardScaler()
        X_catch22 = scaler.fit_transform(X_catch22)

    catch22_names = catch22_i["names"]
    return X_catch22, catch22_names

def matrix_corr(X_catch22, X_words):
    X_corr = np.zeros((np.shape(X_catch22)[1], np.shape(X_words)[1]))
    print(np.shape(X_catch22))
    print(np.shape(X_words))
    for i in range(np.shape(X_catch22)[1]):
        for j in range(np.shape(X_words)[1]):
            X_corr[i, j] = stats.pearsonr(X_catch22[:, i], X_words[:, j])[0]

    return X_corr

# def train_test_(X_train, y_train, X_test):
#     clf = RandomForestClassifier(max_depth=10).fit(X_train, y_train)
#     return clf