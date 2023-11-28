import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from algorithm.datasets import ImageDataset, get_test_loader
from algorithm.classifiers import ResNet34, ResNet50, ResNet101, VGGNet
from algorithm.trainers import CoTeaching, ForwardLossCorrection, JoCoR, O2UNet
from algorithm.eval import eval_metrics

def set_seed(random_seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Parameters:
    random_seed (int): The random seed.
    """
    # Set the random seed for NumPy
    np.random.seed(random_seed)
    # Set the random seed for PyTorch
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True

def percentage_format(x) -> str:
    """
    Format the float number into percentage format.
    """
    return "{:.2f}%".format(x * 100)

def Experiment(seeds, dataset, classifier='ResNet34', robust_method='co_teaching',epochs=300, save_best_model=False, T_estim=False, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Do the experiments for several times and return the results.

    Parameters:
    seeds (list): The list of random seeds.
    dataset (Dataset): The dataset.
    classifier (str, optional): The classifier.
    robust_method (str, optional): The robust method.
    epochs (int, optional): The number of epochs.
    save_best_model (bool, optional): Whether to save the best model.
    T_estim (bool, optional): Whether to estimate the transition matrix.
    device (torch.device, optional): The device to use.
    
    Returns:
    df (DataFrame): The results.
    """
    supported_classifiers = ['ResNet34', 'VGGNet', 'ResNet50', 'ResNet101']
    supported_robust_methods = ['co_teaching', 'loss_correction', 'jocor', 'o2u_net']
    # Check the input parameters, otherwise raise an error.
    if classifier not in supported_classifiers:
        raise ValueError(f"Unsupported classifier: {classifier}. Supported classifiers are: {', '.join(supported_classifiers)}.")

    if robust_method not in supported_robust_methods:
        raise ValueError(f"Unsupported robust method: {robust_method}. Supported robust methods are: {', '.join(supported_robust_methods)}.")
    
    trainer_dict = {'co_teaching': CoTeaching(), 'loss_correction': ForwardLossCorrection(), 'jocor': JoCoR(), 'o2u_net': O2UNet()}
    model_dict = {'ResNet34': ResNet34, 'VGGNet': VGGNet, 'ResNet50': ResNet50, 'ResNet101': ResNet101}
    trainer = trainer_dict[robust_method]
    print(f"We are using {device}.")
    # Initialize the best accuracy and best model
    best_acc = 0.0
    best_model = None
    # Initialize the results list
    results = []
    Ts = []
    for seed in tqdm(seeds):
        # Set the random seed
        set_seed(seed)
        # Load the dataset
        (Xtr, Str), (Xval, Sval), (Xts, Yts) = dataset.load_data(random_state=seed)
        # Fetch necessary information from the dataset
        mean, std = dataset.mean, dataset.std
        T = dataset.T
        if T_estim:
            # If we want to estimate the transition matrix, then force T to be None
            T = None
        input_dim = Xtr.shape[-1]
        # Construct the dataset and data loader
        dataset_tr = ImageDataset(Xtr, Str, mean, std, is_augment=True) # Augment the training data
        dataset_val = ImageDataset(Xval, Sval, mean, std)
        dataset_ts = ImageDataset(Xts, Yts, mean, std)
        tsLoader = get_test_loader(dataset_ts)
        # Branch according to the robust method
        if robust_method == 'co_teaching' or robust_method == 'jocor':
            model1 = model_dict[classifier](input_dim)
            model2 = model_dict[classifier](input_dim)
            trainer.train(model1, model2, dataset_tr, dataset_val, epochs=epochs, T=T, device=device, 
                          show_training_curve=False, show_progress_bar=False, print_early_stopping=False)
            acc_1, precision_1, recall_1, f1_1 = eval_metrics(model1, tsLoader)
            acc_2, precision_2, recall_2, f1_2 = eval_metrics(model2, tsLoader)
            # Select the better model as the final model
            (acc, precision, recall, f1) = (acc_1, precision_1, recall_1, f1_1) if acc_1 > acc_2 else (acc_2, precision_2, recall_2, f1_2)
            if acc > best_acc:
                best_acc = acc
                best_model = model1 if acc_1 > acc_2 else model2
            # Release the memory
            del model1, model2
            torch.cuda.empty_cache()

        elif robust_method == 'loss_correction':
            model = model_dict[classifier](input_dim)
            trainer.train(model, dataset_tr, dataset_val, epochs=epochs, T=T, device=device, 
                          show_training_curve=False, show_progress_bar=False, print_early_stopping=False)
            acc, precision, recall, f1 = eval_metrics(model, tsLoader)
            if acc > best_acc:
                best_acc = acc
                best_model = model
            # Release the memory
            del model
            torch.cuda.empty_cache()
        elif robust_method == 'o2u_net':
            model1, model2 = model_dict[classifier](input_dim), model_dict[classifier](input_dim)
            epochs_0 = int(epochs / 8)
            epochs_1, epochs_2 = int(epochs / 2), int(epochs / 2)
            trainer.train(model1, model2, dataset_tr, dataset_val, 
                          epochs=[epochs_0, epochs_1, epochs_2], T=T, device=device, show_training_curve=False, 
                          show_progress_bar=False, print_training_start=False, print_early_stopping=False)
            acc, precision, recall, f1 = eval_metrics(model2, tsLoader)
            if acc > best_acc:
                best_acc = acc
                best_model = model2
            del model1, model2
            torch.cuda.empty_cache()
        # Store the current metrics into the result dictionary
        result = {
            "Seed": seed,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
        # Append the current result into the results list
        results.append(result)
        if T is None and robust_method == 'loss_correction':
            Ts.append(trainer.T)
    # Convert the results list into a DataFrame
    df = pd.DataFrame(results)
    # Compute the mean and standard deviation of the results
    avg_result = {
        "Seed": "Avg.",
        "Accuracy": df["Accuracy"].mean(),
        "Precision": df["Precision"].mean(),
        "Recall": df["Recall"].mean(),
        "F1": df["F1"].mean()
    }
    std_result = {
        "Seed": "Std.",
        "Accuracy": df["Accuracy"].std(),
        "Precision": df["Precision"].std(),
        "Recall": df["Recall"].std(),
        "F1": df["F1"].std()
    }
    results.append(avg_result)
    results.append(std_result)
    # Append the average result into DataFrame
    df = pd.DataFrame(results)
    # Format the results
    cols = ["Accuracy", "Precision", "Recall", "F1"]
    for col in cols:
        df[col] = df[col].apply(percentage_format)
    if save_best_model:
        # Check if the model hub exists, otherwise create it
        if not os.path.exists("model_hub"):
            os.makedirs("model_hub")
        # Save the model
        save_path = os.path.join("model_hub", f"{dataset.name}_{classifier}_{robust_method}.pth")
        torch.save(best_model.state_dict(), save_path)
    if len(Ts) != 0:
        transition_matrix = sum(Ts) / len(Ts)
        return df, transition_matrix
    else:
        return df