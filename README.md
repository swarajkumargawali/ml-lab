# ML Lab Programs

```python
!pip install --force-reinstall --no-deps https://github.com/swarajkumargawali/ml-lab/archive/refs/heads/main.zip
from mllab import get, nb, list_programs
list_programs()
```

## Step by step: `get(n)`

`get(1)` puts stage 1 of program 1 in the next cell. Press **Shift + Enter** to run it and the next stage appears below, until the program ends.

- If a cell gives an error, the next stage does not appear. Fix it and run it again.
- `get(1, 5)` starts program 1 from stage 5.
- `stop()` stops the chain.

## Full notebook: `nb(n)`

`nb(1)` creates `program1.ipynb` with every stage in its own cell.

| No. | Program | Stages |
|---|---|---|
| 1 | Gaussian Naive Bayes - Iris and credit-card fraud | 13 |
| 2 | Multinomial Naive Bayes - 20 Newsgroups | 7 |
| 3 | Sentiment analysis - Naive Bayes vs Logistic Regression | 8 |
| 4 | Pipeline + GridSearchCV - Scaler, PCA, KNN on Iris | 11 |
| 5 | Random Forest on Iris | 6 |
| 6 | XGBoost on Iris with feature importance | 8 |
| 7 | Customer churn - Random Forest vs XGBoost | 6 |

Programs 1 and 7 need a dataset path pasted inside `r""`.
