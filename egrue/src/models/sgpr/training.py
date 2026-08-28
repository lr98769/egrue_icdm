import numpy as np
import time
import gpflow
import tensorflow as tf
from src.configs.default_configs import device

def model_training_gpr(
    predictors, pred_cols, train_df, valid_df, seed, prop_inducing=0.001):
    # Get data
    train_X, train_y = (
        train_df[predictors].values.astype('float64'), train_df[pred_cols].values.astype('float64'))
    valid_X, valid_y = (
        valid_df[predictors].values.astype('float64'), valid_df[pred_cols].values.astype('float64'))

    rng = np.random.default_rng(seed)
    n_inducing = round(len(train_X)*prop_inducing) # 0.1% of points as inducing points
    print("- Number of Inducing Points:", n_inducing)
    inducing_points = rng.choice(train_X, size=n_inducing, replace=False)
    
    start = time.time()
    with tf.device(f'/gpu:{device.index}'):
        def step_callback(step, variables, values):
            print(f"Step {step} {time.time()-start:.3f} s", end="\r")
        gpr = gpflow.models.SGPR(
            (train_X, train_y),
            kernel=gpflow.kernels.SquaredExponential(),
            inducing_variable=inducing_points,
        )
        opt = gpflow.optimizers.Scipy()
        print(f"Training started...")
        opt.minimize(gpr.training_loss, gpr.trainable_variables, step_callback=step_callback)
    # Train GPR
    # from sklearn.gaussian_process import GaussianProcessRegressor
    # from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    # kernel = DotProduct() + WhiteKernel()
    # gpr = GaussianProcessRegressor(kernel=kernel, copy_X_train=False, random_state=seed)
    # print(f"Training started...")
    # start = time.time()
    # gpr.fit(train_X, train_y)
    
    return gpr