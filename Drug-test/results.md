# Results

## Description
- Using one feature - **DRUG** to tune the best results  

---

## Basic Values
| Metric               | Score                  |
|-----------------------|------------------------|
| Model                | SVM (kernel = RBF)     |
| Cross Validation      | 5-fold                |
| Remission Accuracy    | 0.6429 ± 0.0000       |
| Responder Accuracy    | 0.5857 ± 0.0319       |

---

## Fine-tuning Status
- **Preprocessing** : MinMaxScaler  
- **C** : 1  
- **Gamma** : scale  

---

## Final Accuracy
| Metric          | Value                   |
|-----------------|--------------------------|
| Remission AUC   | 0.4622 ± 0.0877           |
| Responder AUC   | 0.4708 ± 0.0669           |
| Remission       | 0.6429         |
| Responder       | 0.6429        |
