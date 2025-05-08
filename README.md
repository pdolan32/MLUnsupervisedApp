# Unsupervised Machine Learning App Overview:
This interactive Streamlit app offers an intuitive platform for exploring datasets and applying unsupervised machine learning models. The app features dynamic menus and customizable widgets that allow users to tailor their analysis to specific needs and preferences.

The goal of this project is to create an accessible, user-friendly platform that enables individuals to explore unsupervised machine learning for both classification and regression tasks. The app is designed to simplify machine learning workflows, support data-driven decision-making, and encourage experimentation and learning.

With this app, users can:
- Upload their own datasets or choose from built-in sample datasets
- Select from a range of unsupervised learning models, such as Principal Component Analysis (PCA), K-Means Clustering, and Hierarchical Clustering
- Specify and tune model hyperparameters, such as cluster number (k), n_components, and linkage methods
- Visualize model performance with tools like PCA Scatter Plots, Scree Plots, Silhouette Analysis curves, and dendrograms.

Click [HERE]() to access the app online.

# Instructions:

### Prerequisites
Ensure you have the following installed before running this app:
- Python (v. 3.12.7 recommended)
- streamlit (v. 1.44.1)
- pandas (v. 2.2.3)
- numpy (v. 2.2.3)
- scikit-learn (v. 1.6.1)
- matplotlib (v. 3.10.1)
- scipy (v. 1.15.2)

### Running the Application

First, clone the DOLAN-Data-Science-Portfolio repository to create a local copy on your computer: clone the repository by downloading the repository as a ZIP file. Then, extract the downloaded ZIP file to reveal a folder containing all the repository project files. Next, navigate to the MLUnsupervisedApp folder within the extracted content, and upload this folder as your working directory. This folder should include the MLUnsupervisedApp.py file, as well as the README.md file.

To launch the application, use the following command in your terminal:

```bash
streamlit run MLUnsupervisedApp.py
```

The app should open automatically in your default web browser.

# App Features:

This unsupervised machine learning app includes the following unsupervised learning machine models: Principal Component Analysis (PCA), K-Means Clustering, and Hierarchical Clustering. Below is a more detailed explanation of how each of these models are implemented within the app.

## Principal Component Analysis (PCA)

**Principal Component Analysis** is a method used to reduce dimensionality by finding linear combinations of the features that capture maximum variance.

Within the Principal Component Analysis model, users can customize **key hyperparameters**, including:
- n_components, whihc sets the number of new dimensions (features) that PCA will reduce the dataset to: these new dimensions are linear combinations of the original features, ordered by how much variance they explain in the data.

Once these hyperparameters are configured, the model provides the PCA Explained Variance ratio, a Scatter Plot of PCA Scores, a Biplot with Feature Loadings, a Scree plot, a Bar, plot, and a Combined Plot merging the Scree and Bar Plots.

## K-Means Clustering

**K-Means Clustering** is an unsupervised machine learning algorithm used to group data into "k" distinct clusters based on similarity. The algorithm partitions the dataset into k clusters such that each data point belongs to the cluster with the nearest mean (centroid).

Within the K-Means Clustering model, users can customize **key hyperparameters**, including:
- k, which specifies the number of clusters the algorithm will try to find in the dataset.

Once these hyperparameters are configured, the model provides K-Means Cluster Labels in PCA Space, a comparison of these clusters to clusters with true labels, an accuracy score, and an optimal number of clusters determined through both the elbow method and a silhouette score (with accompanying graphics).

## Hierarchical Clustering

**Hierarchical Clustering** is an unsupervised machine learning algorithm used to group similar data points into clusters based on their distance or similarity to one another.

Within the Hierarchical Clustering model, users can customize **key hyperparameters**, including:
- k, which specifies the number of clusters the algorithm will try to find in the dataset.
- the linkage method, which determines how the distance between clusters is calculated when merging them during the clustering process.

Once these hyperparameters are configured, the model provides a Dendrogram (Hierarchical Tree), a table detailing cluster assignments and sizes, a PCA-reduced cluster visualization, and a Silhouette Score analysis (with an accompanying graphic).

## References

Grokking Machine Learning Chapter 2: Types of Machine Learning. Click [HERE](https://github.com/pdolan32/DOLAN-Data-Science-Portfolio/blob/main/MLUnsupervisedApp/Chapter%202.%20Types%20of%20machine%20learning%20-%20Exploring%20Machine%20Learning%20Basics.pdf) to view this reference.

Streamlit Website: Input Widgets. Click [HERE](https://docs.streamlit.io/develop/api-reference/widgets) to access this reference (external website).

Scikit-Learn: Toy Datasets. Click [HERE](https://scikit-learn.org/stable/datasets/toy_dataset.html) to access this reference (external website).

## Visualizations

#### Here are some examples of the visualizations and reports produced by the app when the user chooses to analyze a dataset using the 'Hierarchical Clustering' model.

![dendrogram](https://github.com/user-attachments/assets/c976946b-78dd-4e0f-9c87-a9f53f03d7f7)
![silhouette_curve](https://github.com/user-attachments/assets/344ed012-a9aa-4e68-b3cb-f1f9858a9387)
![pca_analysis](https://github.com/user-attachments/assets/31bd290b-6536-4a62-b5e3-111e5279bd48)









