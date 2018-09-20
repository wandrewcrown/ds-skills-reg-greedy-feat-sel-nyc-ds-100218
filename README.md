
# Feature Selection: An Introduction to Advanced Regression Techniques

In this lab you'll continue to investigate regression models and how models can be over or underfit. We'll also continue to see the role multicollinearity plays in creating regression models. This will then lead us to investigating Regularization techniques such as Lasso and Ridge regression in the next section.  

Today's dataset has a large number of features concerning real estate properties. The final column of this dataset is 'Sales Price' which we would like to create a regression model for. This model could have several purposes. First, if we have data about a property, we would like to know roughly how much the property is apt to sell for. Secondly, we would like to know what features (such as location, number of bedrooms, bathrooms, etc.) are most impactful in determining sales price so that we could prioritize potential home improvements to increase profits.

# Recall the General Outline for Regression

1. Import Data
2. Preprocess
3. Train-Test Split
4. Initialize Model
5. Fit Model
6. Calculate Errors
7. Repeat steps 2-6 with additional feature engineering 

# 1) Import the Data


```python
import pandas as pd
```


```python
df = pd.read_csv('Housing_Kaggle/train.csv')
print(len(df))
df.head()
```

    1460





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



# Feature selection by adding one attribute at a time
Iterate through each feature and calculate the r^2 coefficient for at least 3 different train test split samples. Store these values in a dictionary so that you have
model_scores = {col_name : (avg_r^2_train, avg_r^2_test)} for each column. Then create a dataframe of these values and sort it by the average test score. Preview the top 5 predictive features.




```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
%matplotlib inline
```


```python
X = df.drop('SalePrice', axis=1)
y = df.SalePrice
```


```python
# Your code here
#Iterate through columns
#Perform a regression for 3 different train-test splits
    #You'll have to slightly modify your X_train and X_test becuase they're one dimensional.
    #Do something like this:
    X_train = np.array(X_train[feat]).reshape(-1,1)
#Calculate train / test r^2 for each split
#Store avg train r^2 and avg test r^2 in master dictionary
#Turn master dictionary into dataframe
#Sort dataframe by test r^2
#Preview top 5 predictive features.
```

# Find the second most impactful feature in addition to the first.
Iterate back through all of the column features (except Make note of the most predictive feature from above. Repeat the process to find the second most useful feature in conjunction with the most important feature. **Be sure to include 2 variables in each of your models; one will always be the most predictive from the previous exercise and the second will be some other column.** 


```python
# Your code here
```

# Continue adding features, one at a time and graph train / test accuracy against number of features.
#### Hint: First formalize the previous question as a function. 
The graph should include the number of model feaures on the x axis and the  r^2 values for the train and test splits on the y-axis.


```python
#Your code here
```

# Repeat this process using Mean Squarred Error (MSE) instead of the R^2 Value
Note: MSE should be minimized as opposed to R^2 which we were maximizing.


```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
```


```python
#Your code here
```
