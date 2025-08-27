# Results

## Description
- Using all features to tune the best results  

---

## Basic Values
| Metric               | Score                  |
|-----------------------|------------------------|
| Model                | SVM (kernel = RBF)     |
| Cross Validation      | 8-fold                 |
| Remission Accuracy    | 0.8420 ± 0.0856        |
| Responder Accuracy    | 0.9010 ± 0.1105        |
| Overall Accuracy      | 0.8214 ± 0.1713        |

---

## Fine-tuning Status
- **Preprocessing** : MinMaxScaler  
- **C** : 65  
- **Gamma** : 0.005  

---

## Final Accuracy
| Metric          | Value                   |
|-----------------|--------------------------|
| Remission AUC   | 0.8933 ± 0.1355          |
| Responder AUC   | 0.9333 ± 0.0649          |
| Remission       | 0.9286 (13 / 14)         |
| Responder       | 0.9286 (13 / 14)         |
