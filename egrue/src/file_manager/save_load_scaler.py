import pickle


def load_scaler(fp_downsampled_scaler_file):
    with open(fp_downsampled_scaler_file, 'rb') as handle:
        scaler = pickle.load(handle)
    return scaler


def save_scaler(scaler, fp_downsampled_scaler_file):
    with open(fp_downsampled_scaler_file, 'wb') as handle:
        pickle.dump(scaler, handle)
    return scaler