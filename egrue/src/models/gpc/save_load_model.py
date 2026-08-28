import pickle


def save_gpc(model, fp_model):
    with open(fp_model,'wb') as f:
        pickle.dump(model, f)
    print("Model Saved!")

def load_gpc(fp_model):
    with open(fp_model,'rb') as f:
        model = pickle.load(f)
    return model