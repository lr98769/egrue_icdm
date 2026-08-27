import numpy as np

def model_test_predictions_gpr(
    gpr, df_test, pred_cols, predictors, regressor_label, pred_min, seed):
    df_test = df_test.copy()
  
    test_X, test_y = (
        df_test[predictors].values.astype('float64'), df_test[pred_cols].values.astype('float64'))
    test_y_pred, test_std = gpr.compiled_predict_y(test_X)

    # test_y_pred, test_std = gpr.predict(test_X, return_std=True)
#     print(test_std.shape)
    predicted_colnames = [col+"_gpr"+regressor_label for col in pred_cols]
    std_colnames = [col+"_gpr_std"+regressor_label for col in pred_cols]
    gpr_mean_std_colname = "gpr_std_mean"
    df_test[predicted_colnames] = test_y_pred
    df_test[std_colnames] = test_std
    df_test[gpr_mean_std_colname] = np.mean(test_std, axis=1)
    
    # if df_test['target_index'].dtype != "int64":
    #     df_test['target_index'] = df_test['target_index'].apply(lambda x: eval(x)[pred_min-1])
    
    return df_test