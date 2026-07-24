## Why validation set is required?

- It allows us to choose the best competing model among the group of model

| Model             | Validation RMSE |
| ----------------- | --------------- |
| Linear Regression | 15.2            |
| Random Forest     | 11.4            |
| XGBoost           | **10.8**        |
| CatBoost          | 11.1            |

> Based ont the validation set we choose the best performing model. In this case XGBoost.
