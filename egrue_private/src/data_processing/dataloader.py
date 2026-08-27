from torch.utils.data import DataLoader
from src.data_processing.dataset import TabularDataset

def get_pytorch_split_dict_image(data_dict, batch_size, eval_batch_size, shuffle_train=True, num_workers=4):
    dl_dict = {}
    for split_name in data_dict.keys():
        if "df" in split_name:
            cur_batch_size = batch_size if split_name == "train_df" else eval_batch_size
            cur_shuffle = shuffle_train if split_name == "train_df" else False
            dl_label = split_name[:-1] + "l" # 'train_df' -> 'train_dl'
            ds = data_dict[split_name] # already a dataset
            dl_dict[dl_label] = DataLoader(
                ds, batch_size=cur_batch_size, shuffle=cur_shuffle, pin_memory=True, num_workers=num_workers)
    return dl_dict

def get_tabular_dl_dict(
    train_df, val_df, test_df, feat_cols, target_col, 
    batch_size, eval_batch_size, shuffle_train=True, weight_cols=None):
    train_ds = TabularDataset(df=train_df, feat_cols=feat_cols, target_col=target_col)
    val_ds = TabularDataset(df=val_df, feat_cols=feat_cols, target_col=target_col)
    test_ds = TabularDataset(df=test_df, feat_cols=feat_cols, target_col=target_col)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_dl = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)
    
    return {
        'train_dl':train_dl, 'val_dl':val_dl, 'test_dl':test_dl
    }


def get_tabular_dl_dict_postnet(
    data_dict, batch_size, eval_batch_size, shuffle_train=True, num_workers=4):
    feat_cols = data_dict["feat_cols"]
    target_col = data_dict["target_col"]
    train_ds = TabularDataset(df=data_dict["train_df"], feat_cols=feat_cols, target_col=target_col)
    val_ds = TabularDataset(df=data_dict["val_df"], feat_cols=feat_cols, target_col=target_col)
    test_ds = TabularDataset(df=data_dict["test_df"], feat_cols=feat_cols, target_col=target_col)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, pin_memory=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    return {
        'train_dl':train_dl, 'val_dl':val_dl, 'test_dl':test_dl
    }

