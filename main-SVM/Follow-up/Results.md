# Results - excluding all negatively correlated features

## Basic Values
| Metric                | Value                       |
|------------------------|-----------------------------|
| Model                 | SVM (kernel = RBF)          |
| Cross Validation      | 8-fold                      |
| Remission Accuracy    | 0.9167 ± 0.0985             |
| Responder Accuracy    | 0.8872 ± 0.1030             |
| Accuracy Score        | 0.8214 ± 0.1185             |

---

## Fine-tuning Status
| Parameter     | Value          |
|---------------|----------------|
| Preprocessing | MinMaxScaler   |
| C             | 180            |
| Gamma         | 0.008          |

---

## Excluding Negatively Dataset vs. Entire Dataset
| Metric                        | Excluding Negatively | Entire Dataset | Gain |
|-------------------------------|----------------------|----------------|------|
| Accuracy Score                | 0.8214               | 0.8214         | –    |
| Std. Dev. of Accuracy         | 0.1185               | 0.1713         | +45% |

---
<img width="1832" height="996" alt="image" src="https://github.com/user-attachments/assets/b5a59b5f-c3a9-459b-bb9d-9647b0a13633" />
