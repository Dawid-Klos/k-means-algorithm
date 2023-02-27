import pandas as pd
import k_means as km
import matplotlib.pyplot as plt

######################
###  Informations  ###
######################
#
# !!! READ BEFORE RUNNING!!!
#
# To run the program simply execute the main file - main.py
# While running the program check the console for information about performed operations.
# When a new window with a graph appears to proceed you need to close it by clicking on the close button.
# Keep closing the window until the end of the program.
# In the meantime you will see information being printed in the console.
# Once the best fit centroids are found, the program finish working and you can see the performance results in the console.
#
#
######################
### Pre-processing ###
######################

# import data from a csv file
customers_df = pd.read_csv('mall_customers.csv')

# drop redundant columns
customers_df = customers_df.drop(columns=['CustomerID', 'Gender', 'Age'])

# turn dataframe into a list
data = customers_df.values.tolist()

########################
###    Algorithm     ###
########################

# 1. Predict number of clusters by using the Elbow method
# Check for best fit between 0 and 10 clusters
print('Running the Elbow Method... \n')
wcss_list = []
for i in range(0, 10):
    print(f'Number of clusters: {i+1}')
    kmeans = km.Kmeans(k=i+1, max_iter=30, dataset=data, print_info=False)
    kmeans.start_clustering()
    wcss = kmeans.check_accuracy()
    wcss_list.append(wcss)

plt.plot(range(10), wcss_list, linestyle="-", marker="o", color="blue")
plt.title(f'The Elbow Method')
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

print('\nThe Elbow is not clear enough, therefore perform closer look at values around the Elbow\n')

# Check for best fit between 2 and 6 clusters
wcss_list = []
for i in range(2, 7):
    print(f'Number of clusters: {i}')
    kmeans = km.Kmeans(k=i, max_iter=50, dataset=data, print_info=False)
    kmeans.start_clustering()
    wcss = kmeans.check_accuracy()
    wcss_list.append(wcss)

plt.plot([2, 3, 4, 5, 6], wcss_list,
         linestyle="-", marker="o", color="blue")
plt.title(f'The Elbow Method - closer look')
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
print('\n--> End of the Elbow Method\n')
print('################################################################\n')
print('\n--> 4 clusters seems to be a good choice\n\n')

# 2. Initialize Kmeans class

# --> k = number of clusters
# --> max_iter = number of iterations the algorithm will use to fit the data into clusters
# --> data = dataset after preprocessing as a list
# --> print_info = if True, print information about performed operations to the console and as a graph

kmeans = km.Kmeans(k=4, max_iter=40, dataset=data, print_info=True)


# 3. Run algorithm
kmeans.start_clustering()


# 4. Analize performance by calculating WCSS
kmeans.check_accuracy()
