# Robust Trainers for Noisy Labels

## Introduction
This project is an experimental repository focusing on dealing with datasets containing a high level of noisy labels (50% and above). This repository features experiments conducted on the `FashionMNIST` and `CIFAR` datasets using the `ResNet34` as the baseline classifier.

The repository explores various training strategies (`trainers`), including `ForwardLossCorrection`, `CoTeaching`, `JoCoR`, and `O2UNet`. Specifically, for datasets with unknown transition matrices, `DualT` is employed as the Transition Matrix Estimator. Given the computational complexity and practical performance considerations, our experiments primarily focus on `ForwardLossCorrection` and `CoTeaching`. We conducted multiple experiments with different random seeds to compare these two methods.

Initial explorations on `FashionMNIST0.5` with `JoCoR` and `O2UNet` have shown promising results. This repository serves as a resource for those interested in robust machine learning techniques under challenging conditions of high label noise.

A brief pipeline: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/XavierSpycy/Robust-Trainers-for-Noisy-Labels/blob/main/notebook.ipynb)

## Experimental Setup
- Datasets (3 classes: 0, 1, 2; instead of 10): 
    - FashionMNIST with Known Flip Rate
        - Noise Level: 0.5
        <div align="center">
            <h5>Transition Matrix of FashionMNIST0.5</h5>
            <table>
            <tr><td>0.5</td><td>0.2</td><td>0.3</td></tr>
            <tr><td>0.3</td><td>0.5</td><td>0.2</td></tr>
            <tr><td>0.2</td><td>0.3</td><td>0.5</td></tr>
            </table>
        </div>
        
        <p align="center">
            <img src="figures/tSNE-1.png">
            <br>
            Noisy Data
        </p>
        <p align="center">
            <img src="figures/tSNE-2.png">
            <br>
            Clean Data
        </p>
        
        - Noise Level: 0.6
        <div align="center">
            <h5>Transition Matrix of FashionMNIST0.6</h5>
            <table>
            <tr><td>0.4</td><td>0.3</td><td>0.3</td></tr>
            <tr><td>0.3</td><td>0.4</td><td>0.3</td></tr>
            <tr><td>0.3</td><td>0.3</td><td>0.4</td></tr>
            </table>
        </div>

        <p align="center">
            <img src="figures/aug-1.png">
            <br>
            Samples in FashionMNIST0.6
        </p>

    - CIFAR with Unknown Flip Rate
        <p align="center">
            <img src="figures/cifar.png">
            <br>
            Samples in CIFAR
        </p>

- Base Classifier:
    - ResNet-34
    
- Basic Robust Method(s):
    - Data Augmentation

- Robust Trainers:
    - Loss correction: `ForwardLossCorrection`
        - Includes: `SymmetricCrossEntropyLoss`
    - Multi-network learning: `CoTeaching`
    - Multi-network learning: `JoCoR`
    - Multi-round learning: `O2UNet` 

- Transition Matrix Estimator:
    - `Dual-T`


## Results
### Loss value trends
According to the the loss trends, we find that our robust trainers may also act as regularizers to avoid overfitting.

- `ForwardLossCorrection`
<p align="center">
    <img src="figures/loss_correction_trend.png">
    <br>
    Loss Trend
</p>

-  `CoTeaching`
<p align="center">
    <img src="figures/co_teaching_trend.png">
    <br>
    Loss Trend
</p>

### Performance
We have conducted a series of experiments utilizing 10 distinct random seeds to evaluate the performance of `ForwardLossCorrection` and `CoTeaching`. Below is a detailed comparison of their performances.

<div align="center">
<table>
    <h4>Peformance Comparison</h4>
    <tr>
        <td rowspan="2" align='center'>Dataset</td>
        <td rowspan="2">Metrics</td>
        <td colspan="2" align='center'>Robust Trainer</td>
    </tr>
    <tr>
        <td align='center'>ForwardLossCorrection</td>
        <td align='center'>CoTeaching</td>
    </tr>
    <tr>
        <td rowspan="4" align='center'>FashionMNIST0.5</td>
        <td align='center'>Accuracy</td>
        <td align='center'>77.47%(&plusmn; 6.33%)</td>
        <td align='center'>90.33%(&plusmn; 3.34%)</td>
    </tr>
    <tr>
        <td align='center'>Precision</td>
        <td align='center'>78.87%(&plusmn; 5.75%)</td>
        <td align='center'>90.93%(&plusmn; 2.49%)</td>
    </tr>
    <tr>
        <td align='center'>Recall</td>
        <td align='center'>77.47%(&plusmn; 6.33%)</td>
        <td align='center'>90.33%(&plusmn; 3.34%)</td>
    </tr>
    <tr>
        <td align='center'>F1 Score</td>
        <td align='center'>77.53%(&plusmn; 6.54%)</td>
        <td align='center'>90.29%(&plusmn; 3.46%)</td>
    </tr>
    <tr>
        <td rowspan="4" align='center'>FashionMNIST0.6</td>
        <td align='center'>Accuracy</td>
        <td align='center'>77.05%(&plusmn; 6.61%)</td>
        <td align='center'>80.25%(&plusmn; 12.44%)</td>
    </tr>
    <tr>
        <td align='center'>Precision</td>
        <td align='center'>80.08%(&plusmn; 3.64%)</td>
        <td align='center'>75.28%(&plusmn; 20.81%)</td>
    </tr>
    <tr>
        <td align='center'>Recall</td>
        <td align='center'>77.05%(&plusmn; 6.61%)</td>
        <td align='center'>80.25%(&plusmn; 12.44%)</td>
    </tr>
    <tr>
        <td align='center'>F1 Score</td>
        <td align='center'>76.27%(&plusmn; 8.55%)</td>
        <td align='center'>76.92%(&plusmn; 17.83%)</td>
    </tr>
    <tr>
        <td rowspan="4" align='center'>CIFAR</td>
        <td align='center'>Accuracy</td>
        <td align='center'>49.81%(&plusmn; 12.58%)</td>
        <td align='center'>47.28%(&plusmn; 4.09%)</td>
    </tr>
    <tr>
        <td align='center'>Precision</td>
        <td align='center'>50.11%(&plusmn; 12.06%)</td>
        <td align='center'>33.41%(&plusmn; 3.73%)</td>
    </tr>
    <tr>
        <td align='center'>Recall</td>
        <td align='center'>49.81%(&plusmn; 12.58%)</td>
        <td align='center'>47.28%(&plusmn; 4.09%)</td>
    </tr>
    <tr>
        <td align='center'>F1 Score</td>
        <td align='center'>49.09%(&plusmn; 12.27%)</td>
        <td align='center'>38.04%(&plusmn; 3.69%)</td>
    </tr>
    </table>
</div>

It becomes evident from our analysis that `CoTeaching` exhibits superior performance under conditions of low noise. However, as the noise level escalates, `ForwardLossCorrection` demonstrates enhanced robustness, outperforming `CoTeaching`.

In our preliminary experiments, both `JoCoR` and `O2UNet` showed promising results on the FashionMNIST0.5 dataset. Nevertheless, due to the substantial computational demands and the marginal improvements they offered over `CoTeaching`, we decided not to proceed with extensive experimentation on these methods.

### Estimation of Transition Matrix
- Estimation on FashionMNIST05
<div align="center">
    <h5>Estimated Transition Matrix of FashionMNIST0.5</h5>
    <table>
    <tr><td>0.473</td><td>0.209</td><td>0.309</td></tr>
    <tr><td>0.306</td><td>0.485</td><td>0.232</td></tr>
    <tr><td>0.221</td><td>0.306</td><td>0.460</td></tr>
    </table>
</div>

- Estimation on FashoinMNIST06
<div align="center">
    <h5>Estimated Transition Matrix of FashionMNIST0.6</h5>
    <table>
    <tr><td>0.407</td><td>0.295</td><td>0.298</td></tr>
    <tr><td>0.297</td><td>0.394</td><td>0.308</td></tr>
    <tr><td>0.301</td><td>0.310</td><td>0.388</td></tr>
    </table>
</div>

- Estimation on CIFAR
<div align="center">
    <h5>Estimated Transition Matrix of CIFAR</h5>
    <table>
    <tr><td>0.365</td><td>0.332</td><td>0.311</td></tr>
    <tr><td>0.337</td><td>0.368</td><td>0.315</td></tr>
    <tr><td>0.298</td><td>0.300</td><td>0.374</td></tr>
    </table>
</div>