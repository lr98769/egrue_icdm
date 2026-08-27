from src.misc import set_seed_tf

import numpy as np
import tensorflow as tf

def infernoise_test_predictions(ae_predictor, test_df, inputs, outputs, seed, T, stddev, regressor_label, time_run=True):
    test_df = test_df.copy()
    # Process data
    test_X, test_y = (test_df[inputs].values.astype('float32'), test_df[outputs].values.astype('float32'))

    set_seed_tf(seed)

    # Define model
    input_layer = tf.keras.Input(shape=(len(inputs),))
    gaussian_noise_layer = tf.keras.layers.GaussianNoise(stddev, seed=seed)
    x = ae_predictor.encoder(input_layer)
    x = gaussian_noise_layer(x)
    x = ae_predictor.regressor.layers[-1](x) # To ignore dropout layer
    infernoise_model = tf.keras.Model(inputs=input_layer, outputs=x, name="infernoise_model")

    # For Infer Noise
    # - Sample with infer noise
    test_y_sample_preds = [] # T, num samples, output classes
    for i in range(T):
        test_y_pred = infernoise_model(test_X, training=True)
        # print(test_y_pred[:5])
        test_y_sample_preds.append(test_y_pred)

    test_y_sample_preds = np.array(test_y_sample_preds)
    test_y_sample_preds =  test_y_sample_preds.transpose((1, 2, 0)) # num samples, predicted features, T

    # - Get mean prediction 
    # num samples, predicted features
    test_y_mean_pred = np.mean(test_y_sample_preds, axis=-1)
    predicted_infernoise_colnames = [col + "_infernoise"+regressor_label for col in outputs]
    test_df[predicted_infernoise_colnames] = test_y_mean_pred

    # - Get loss for each output
    test_y_mc_loss = mae_fn(y_true=test_y, y_pred=test_y_mean_pred)
    test_df["infernoise_mae"] = test_y_mc_loss

    # - Uncertainty score (for regression)
    #   - Calculate std of predictions
    test_y_sd_pred = np.mean(np.std(test_y_sample_preds, axis=-1, ddof=1), axis=-1)
    test_df["infernoise_uncertainty"] = test_y_sd_pred

    return test_df


def mae_fn(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred), axis=-1)
