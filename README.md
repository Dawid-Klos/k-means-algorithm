## K-means - clustering algorithm implementation

This is my implementation of the K-means clustering algorithm for data classification in Python. 


### Dataset

The dataset includes following attributes and types:
- CustomerID - int64
- Gender - object
- Age - int64
- Annual Income (k$) -int64
- Spending Score (1-100) – int64

The number of records in the chosen dataset is equal to 200. The data set was obtained from
[kaggle.com](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)


### Pre-processing

The pre-processing steps for a given dataset included dropping redundant columns and turning the pandas dataframe into list. If you wish to give a try on a different dataset then different steps would need to be done. The K-means class takes dataset as a list with tuples. Therefore, each tuple is a point on the graph [x, y].

### Other Libraries

To run the program make sure you've got the following libraries installed using pip:

- Pandas
- Numpy
- Matplotlib.pyplot
- Random

### Running

To run the program simply execute the main.py file. Then watch the output in the console for a glimpsy what is being undertaken at the moment. The program also opens a new window for each new data plot. To proceed you need to close that window.

