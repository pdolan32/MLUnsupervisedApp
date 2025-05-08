# Import the required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Write in the title of the app
st.title('Unsupervised Machine Learning Dataset Analysis')

# Write in the preliminary app instructions
st.write('To begin, consult the sidebar to either upload a dataset or choose from a sample of datasets.')

# Creates a sidebar for user input
# Using the radio widget, the user is presented with the option to either upload a dataset or choose a sample dataset
option = st.sidebar.radio('Choose Dataset Option', ('Upload Your Own', 'Use Sample Dataset'))

target_column = None
df = None

if option == 'Upload Your Own': # If the user chooses to upload their own dataset
    uploaded_file = st.sidebar.file_uploader("Upload a .csv file", type='csv') # Creates a file uploader widget in the sidebar of the Streamlit app.

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) # This checks whether a file was uploaded: if yes (uploaded_file is not None), it reads the uploaded .csv file using Pandas
        st.write("In order to analyze your uploaded dataset, please choose an unsupervised machine learning model from the sidebar.")
        st.write("Please use the tools and widgets in the sidebar to prepare your dataset for analysis. There is a cleaned dataset at the bottom that will be used in the analysis and will reflect the results of your data preprocessing choices.")
        st.sidebar.subheader("Data Preprocessing Options")
        
        st.subheader("Uploaded Dataset:")
        st.write(df.head()) # The first five rows of the uploaded dataset are displayed

        st.subheader("Summary Statistics:")
        st.write(df.describe()) # The summary statistics of the uploaded dataset are displayed

        if 'df_clean' not in st.session_state: 
            st.session_state.df_clean = df.copy() # Creates a copy of the original DataFrame df and stores this copy in st.session_state.df_clean so that it persists across user interactions.

        df_clean = st.session_state.df_clean # retrieves the persistent version of the cleaned dataset from session_state and assigns it to the local variable df_clean.

        if st.sidebar.checkbox("Does your dataset include a target column?"):
            # Displays a dropdown menu (selectbox) with all column names in the DataFrame df; lets the user choose one as the target column
            target_column = st.sidebar.selectbox("Select target column:", df.columns) 
            if target_column:
                target_names = df[target_column].unique() # df[target_column].unique() gets all unique values in that column 
                target_names.sort()

        cols_to_drop = st.sidebar.multiselect( # Adds a multiselect widget in the sidebar, allowing the user to choose multiple columns from the original DataFrame df to drop.
            "Select columns to drop:", df.columns, default=[]
        )

        # Apply drops only if user selected columns and they exist in the current df_clean
        if cols_to_drop:
            # Only drop if those columns haven't already been dropped
            remaining_cols = set(df_clean.columns)
            valid_cols_to_drop = [col for col in cols_to_drop if col in remaining_cols]
            if valid_cols_to_drop: # Drops only the valid, remaining columns from df_clean
                df_clean.drop(columns=valid_cols_to_drop, inplace=True)
                st.session_state.df_clean = df_clean  # Save the updated df_clean
                st.success(f"Dropped columns: {valid_cols_to_drop}")

        if st.sidebar.button("Drop Rows with Missing Values"):
            df_clean = df_clean.dropna() # Drop rows with missing values
            st.session_state.df_clean = df_clean  # Update session state
            st.success("Rows with missing values have been removed.")

        # allows users to apply one-hot encoding to any categorical variables in their dataset
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist() # Identifies all columns in df_clean that are of type 'object' or 'category'
        if st.sidebar.checkbox("Apply one-hot encoding to categorical columns?"):
            # Applies one-hot encoding using pd.get_dummies()
            if categorical_cols:
                st.write(f"Encoding the following categorical columns: {categorical_cols}")
                df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True) # avoids multicollinearity by dropping the first category in each encoded column.
                st.session_state.df_clean = df_clean  # Save the updated df_clean
            else:
                st.write("No categorical columns detected for encoding.")

        # Final output
        st.subheader("Cleaned Dataset Preview:")
        st.write(df_clean.head()) # Displays the first 5 rows of the cleaned dataset (df_clean)
        st.markdown(f"**Total rows in cleaned dataset:** {df_clean.shape[0]}")

        # Updates the main df variable to now reference the cleaned version of the data.
        df = df_clean

else: # If the user selected not to upload their own data, this line displays a dropdown in the sidebar to let them pick from three built-in sample datasets.
    dataset_option = st.sidebar.selectbox('Choose Sample Dataset', ('Breast Cancer', 'Iris', 'Wine'))

    # Depending on which dataset the user selects, it loads the corresponding dataset from sklearn.datasets.
    if dataset_option == 'Breast Cancer':
        data = load_breast_cancer()
        st.write('This is the Breast Cancer dataset.' \
        ' This dataset is a well-known dataset used for binary classification, often to test models that distinguish between benign and malignant tumors: The target value is the diagnosis, with 0 = benign and 1 = malignant.')
        st.write('In order to analyze this dataset, please choose an unsupervised machine learning model from the sidebar.')
    elif dataset_option == 'Iris':
        data = load_iris()
        st.write('This is the Iris dataset.' \
        ' The Iris dataset contains information about 150 iris flowers from 3 different species.' \
        ' Each sample represents one flower, and the goal is to classify the species of the flower based on 4 features.' \
        ' The species of the Iris flower is categorical (0, 1, 2): Iris setosa, Iris versicolor, Iris virginica, respectively.')
        st.write('In order to analyze this dataset, please choose an unsupervised machine learning model from the sidebar.')
    elif dataset_option == 'Wine':
        data= load_wine()
        st.write('This is the Wine dataset.' \
        ' This dataset is a classic multiclass classification dataset often used for testing machine learning models. It contains the chemical analysis of wines grown in the same region in Italy but derived from three different cultivars (classes).')
        st.write('In order to analyze this dataset, please choose an unsupervised machine learning model from the sidebar.')

    df = pd.DataFrame(data.data, columns=data.feature_names) # Converts the feature data into a pandas DataFrame using the provided feature names as column headers.
    if hasattr(data, 'target'): # If the dataset has a .target attribute (which all three do), it adds it to the DataFrame as a 'target' column. It also sets:
        df['target'] = data.target
        target_column = 'target' # target_column to 'target' (to be used elsewhere in the app),
        target_names = data.target_names if hasattr(data, 'target_names') else np.unique(data.target) # target_names to class names if available (e.g., 'malignant', 'benign'), or otherwise just the unique target values.
    else: # If the dataset doesn't have a target, this avoids errors by setting target_names to None.
        target_names = None

    # Displays the dataset name, shows the first few rows of the dataset, and prints summary statistics for numerical columns.
    st.subheader(f"Sample Dataset: {dataset_option}")
    st.write(df.head())
    st.subheader("Summary Statistics:")
    st.write(df.describe())

if df is not None: # Checks whether a DataFrame (df) has been successfully loaded—either through a user upload or a sample dataset selection. 
                   #If no dataset is loaded, the following code won’t run.

    if target_column and target_column in df.columns: # Checks two things:
                                                      # that a target column has been defined (i.e. target_column is not None).
                                                      # that the specified target_column actually exists in the DataFrame df.
        X = df.drop(columns=[target_column]) # All columns except the target column. This is the feature matrix.
        y = df[target_column].values # The values in the target column. This is the label/target vector.
    else: # If no valid target column was provided, it treats the entire dataset as X (all features)
        X = df.copy()
        y = None
        target_names = None

    # Adds a dropdown in the sidebar for the user to select which analysis/model they want to run.
    model_option = st.sidebar.selectbox('Choose a Model', ('None', 'Principal Component Analysis', 'K-Means Clustering', 'Hierarchical Clustering'))

    if model_option == 'Principal Component Analysis': # Only run this block if the user selected Principal Component Analysis in the sidebar.
        
        st.subheader('Principal Component Analysis')
        st.write('The unsupervised learning model you have chosen is: Principal Component Analysis (PCA). PCA is a method used to reduce dimensionality by finding linear combinations of the features that capture maximum variance.')
        st.write('Note: since PCA is sensitive to the scale of the variables, the data is centered and scaled')

        # Uses StandardScaler to center and scale the features
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # The user can choose how many principal components to extract, from 2 up to a maximum of 15 or however many features the dataset has.
        # Starts with a default of 2 for visualization purposes.
        max_components = min(X.shape[1], 15)
        st.sidebar.subheader('PCA Options')
        n_components = st.sidebar.slider('Number of Principal Components', 2, max_components, 2)

        # Reduces the standardized feature data into 'n_components' principal components. X_pca is now the transformed dataset in the lower-dimensional space.
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_std)

        # Display the Explained Variance Ratio
        st.subheader('Explained Variance Ratio')
        st.write('The explained variance ratio details the proportion of variance explained by each component.')
        st.write('Try using the slider on the sidebar to adjust the amount of components included in the analysis to see how much of the total variance each component explains.' \
        ' Adjusting the amount of components will also influence some of the visualizations (see below).')
        explained_variance = pca.explained_variance_ratio_ # How much variance each principal component explains.
        cumulative_variance = np.cumsum(explained_variance) # Running total of variance explained by the components.

        # Outputs a table showing how much each component contributes to explaining the dataset’s variance.
        explained_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
            'Explained Variance (%)': (explained_variance * 100).round(2),
            'Cumulative Variance (%)': (cumulative_variance * 100).round(2)
        }, index=np.arange(1, len(explained_variance) + 1))

        # Display the table in Streamlit
        st.subheader("PCA Explained Variance")
        st.dataframe(explained_df)

        # Prepares feature names for any further plotting
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"Feature {i}" for i in range(X.shape[1])]
        
        unique_labels = np.unique(y) if y is not None else []
        n_classes = len(unique_labels)
        color_map = plt.cm.get_cmap('tab10', n_classes) # If y (target labels) exist, it sets up a color map for up to 10 unique target values using matplotlib's 'tab10' palette.

        # --- 1. PCA 2D Scatter Plot ---
        st.subheader('Scatter Plot of PCA Scores')
        st.write('This scatter plot plots the data points in the new coordinate system defined by the first two principal components and helps visualize how the observations are spread out and whether distinct groups exist.')
        st.write('Note: since the data is projected into two dimensions, only the first two principcal components are visualized here.')
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        color_map = plt.get_cmap('tab10')
        if y is not None: # Checks if a target variable y (e.g., class labels) is available. If so, plot points with different colors based on their class.
            for i, label in enumerate(unique_labels): # Loops through each unique class label.
                label_name = str(target_names[label]) if target_names is not None else str(label)
                ax1.scatter(X_pca[y == label, 0], X_pca[y == label, 1], # Plots the PCA points that belong to a specific class
                            color=color_map(i), alpha=0.7, edgecolor='k', s=60, label=label_name)
        else: # If no y (target labels), it plots all points in gray without label-based separation.
            ax1.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.7, edgecolor='k', s=60)
        # Below code adds axis labels and a title to the plot.
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('PCA: 2D Projection')
        ax1.legend(loc='best') # Adds a legend with class names (only if y is provided).
        ax1.grid(True) # Adds a grid to make the plot easier to read.
        st.pyplot(fig1) # Renders the figure in the Streamlit app.

        # --- 2. PCA Biplot with Feature Loadings ---
        st.subheader('Biplot with Feature Loadings')
        st.write('This biplot overlays the original feature vectors (loadings) on the scatter plot to interpret the direction and contribution of each feature in the reduced space.' \
        ' The loading vectors indicate the direction in which the original features contribute most to the variance captured by the principal components.')
        loadings = pca.components_.T # Retrieves the loadings (a.k.a. component weights), which describe the contribution of each original feature to each principal component.
        scaling_factor = 50.0
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        color_map = plt.get_cmap('tab10')
        if y is not None: # If class labels (y) are available: plots the PCA-transformed data, coloring each class differently.
            for i, label in enumerate(unique_labels):
                label_name = str(target_names[label]) if target_names is not None else str(label)
                ax2.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                            color=color_map(i), alpha=0.7, edgecolor='k', s=60, label=label_name)
        else: # If no labels are provided, it plots all points in gray.
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.7, edgecolor='k', s=60)
        for i, feature in enumerate(feature_names): # Loops through each original feature to plot its vector (arrow) on the PCA plot.
            ax2.arrow(0, 0, scaling_factor * loadings[i, 0], scaling_factor * loadings[i, 1], # Draws an arrow from the origin (0, 0) to the scaled coordinates in PCA space.
                    color='r', width=0.02, head_width=0.1)
            ax2.text(scaling_factor * loadings[i, 0] * 1.1, scaling_factor * loadings[i, 1] * 1.1, # Adds a text label for each arrow at the tip.
                    feature, color='r', ha='center', va='center')
        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.set_title('Biplot: PCA Scores and Feature Loadings')
        ax2.legend(loc='best') # Adds a legend if y is present (class labels).
        ax2.grid(True)
        st.pyplot(fig2) # Renders the figure in the Streamlit app.

        # --- 3. Scree Plot of Cumulative Explained Variance ---
        st.subheader('Scree Plot')
        st.write('The scree plot displays the cumulative proportion of variance explained by the principal components: it is a valuable tool for deciding how many components to retain for further analysis.')
        st.write('Note: adjusting the sidebar slider to increase/decrease the number of principal components will expand/reduce the following plots, respectively.')
        pca_full = PCA(n_components=n_components).fit(X_std) # Creates and fits a PCA model using up to the specified amount of components)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_) # Calculates the cumulative sum of the explained variance ratio
        fig3, ax3 = plt.subplots(figsize=(8, 6)) # Creates a new figure and axis for the line chart.
        ax3.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o') # Plots the cumulative variance explained versus the number of components.
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Cumulative Explained Variance')
        ax3.set_title('PCA Variance Explained')
        ax3.set_xticks(range(1, len(cumulative_variance) + 1)) # Ensures the x-axis only shows integer component counts (e.g., 1 to 15).
        ax3.grid(True)
        st.pyplot(fig3) # Renders the plot within the Streamlit app

        st.write('Elbow Method: Look for the "elbow" in the plot - beyond this point, additional components contribute only marginal gains in explained variance.')

        # --- 4. Bar Plot of Individual Variance ---
        st.subheader('Bar Plot')
        st.write('The Bar Plot help to visualize the variance explained by each component.')
        fig4, ax4 = plt.subplots(figsize=(8, 6)) # Creates a new figure and axis for the bar plot.
        components = range(1, len(pca_full.explained_variance_ratio_) + 1) # Creates a range of component numbers (1-based) to use as x-axis labels.
        ax4.bar(components, pca_full.explained_variance_ratio_, alpha=0.7, color='teal') # Draws a bar for each principal component, where the height represents how much variance that component explains.
        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Variance Explained')
        ax4.set_title('Variance Explained by Each Principal Component')
        ax4.set_xticks(components) # Ensures x-axis ticks match the component numbers exactly.
        ax4.grid(True, axis='y')
        st.pyplot(fig4) # Renders the chart within the Streamlit app

        # --- 5. Combined Plot ---
        st.subheader('Combined Plot')
        st.write('This Combined Plot combines the scree and bar plots.')
        explained = pca_full.explained_variance_ratio_ * 100 # Converts the explained variance from a ratio to a percentage.
        components = np.arange(1, len(explained) + 1) # Generates component numbers starting at 1 (for display and labeling).
        cumulative = np.cumsum(explained) # Calculates the cumulative percentage of variance explained.

        fig5, ax5 = plt.subplots(figsize=(8, 6)) # Creates the primary axis and draws a bar for each component showing the % of variance it individually explains.
        bar_color = 'steelblue'
        # Labels the x-axis and left y-axis, matching their colors for clarity.
        ax5.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')
        ax5.set_xlabel('Principal Component')
        ax5.set_ylabel('Individual Variance Explained (%)', color=bar_color)
        ax5.tick_params(axis='y', labelcolor=bar_color)
        ax5.set_xticks(components)
        ax5.set_xticklabels([f"PC{i}" for i in components])
        for i, v in enumerate(explained): # Adds text labels above each bar showing the exact variance %.
            ax5.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

        ax6 = ax5.twinx() # Adds a second y-axis that shares the same x-axis, and plots the cumulative variance as a line.
        line_color = 'crimson'
        # Sets and styles the right y-axis, ensuring it scales from 0% to 100%.
        ax6.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance')
        ax6.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
        ax6.tick_params(axis='y', labelcolor=line_color)
        ax6.set_ylim(0, 100)

        # Combines both legends (from bar and line plots) into one unified legend placed neatly on the side.
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax6.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

        # Adds a descriptive title with spacing above and renders the final chart in Streamlit.
        plt.title('PCA: Variance Explained', pad=20)
        st.pyplot(fig5)

    if model_option == 'K-Means Clustering': # Only run this block if the user selected K-Means Clustering in the sidebar.

        st.subheader('K-Means Clustering')
        st.write('The unsupervised learning model you have chosen is: K-Means Clustering. K-Means Clustering is an unsupervised machine learning algorithm used to group data into "k" distinct clusters based on similarity. The algorithm partitions the dataset into k clusters such that each data point belongs to the cluster with the nearest mean (centroid).')
        st.write('While reading through the model results, consult the sidebar to try changing the number of clusters and observe how the visuals and metrics change.')
        st.write('Note: since K-Means Clustering relies on distance calculations and can be biased by the scale of features, the data is centered and scaled')

        # Uses StandardScaler to center and scale the features: this is crucial for K-Means Clustering since it uses Euclidean distance.
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Adds a sidebar slider so the user can control how many clusters the K-Means algorithm tries to find.
        st.sidebar.subheader("Clustering Options")
        k = st.sidebar.slider('Select number of clusters (k)', min_value=2, max_value=10, value=2, step=1)
        # Fits the K-Means model and predicts cluster labels.
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_std)

        # Reduces high-dimensional standardized data to 2 components for plotting.
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)

        # --- Plot 1: K-Means Cluster Labels in PCA Space ---
        st.subheader('K-Means Cluster Labels in PCA Space')
        st.write('KMeans clustering partitions the data into k clusters by iteratively assigning points to the nearest cluster centroid and then updating the centroids based on the mean of the clusters.')
        st.write('Visualizations reveal how well K-Means Clustering has partitioned the data; however, many datasets are high-dimensional, so the datasets are first reduced to 2 dimensions using PCA for visualization. Then, the PCA scores are plotted with colors corresponding to the cluster assignments.')
        fig1, ax1 = plt.subplots(figsize=(8, 6)) # Plots the K-Means cluster assignments in PCA space.

        cluster_labels = np.unique(clusters)
        n_clusters = len(cluster_labels)
        # Each cluster is shown in a different color using matplotlib’s tab10 colormap.
        color_map = plt.cm.get_cmap('tab10')

        for i in cluster_labels: # plots each cluster group in a 2D PCA space
            ax1.scatter(
                X_pca[clusters == i, 0], X_pca[clusters == i, 1],
                color=color_map(i), alpha=0.7, edgecolor='k', s=60,
                label=f'Cluster {i}'
            )
        # Includes legend and labeled axes.
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('KMeans Clustering: 2D PCA Projection')
        ax1.legend(loc='best')
        ax1.grid(True)
        st.pyplot(fig1)

        # --- Plot 2: True Labels (if provided) in PCA Space ---
        st.subheader('Comparing Clusters with True Labels')
        st.write('Even though K-Means Clustering is an unsupervised machine learning model, its cluster assignments can be compared with the actual labels to gauge performance.')
        fig2, ax2 = plt.subplots(figsize=(8, 6)) # Plots the true class labels 

        if y is not None: # Uses the same PCA projection to allow easy visual comparison between predicted clusters and true labels.
            true_labels = np.unique(y)
            n_classes = len(true_labels)
            color_map = plt.cm.get_cmap('tab10')

            for i in true_labels: # creates one scatterplot layer per class, so that the PCA projection is color-coded by the actual class labels, and the plot legend reflects readable names.
                label_name = str(target_names[i]) if target_names is not None else str(i)
                ax2.scatter(
                    X_pca[y == i, 0], X_pca[y == i, 1],
                    color=color_map(i), alpha=0.7, edgecolor='k', s=60,
                    label=label_name
                )
            ax2.set_title('True Labels: 2D PCA Projection')
        else:
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.7, edgecolor='k', s=60)
            ax2.set_title('True Labels (Unavailable)')

        # Includes legend and labeled axes.
        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.legend(loc='best')
        ax2.grid(True)
        st.pyplot(fig2)

        # --- Accuracy Score ---
        st.subheader('Accuracy Score')
        st.write('Although K-Means Clustering is an unsupervised machine learning model, the accuracy score assesses how well the clusters match the true labels.')
        st.write('Note: Since KMeans labels are arbitrary (e.g., 0 and 1) and may not match the true labels directly, accuracy for both the original labels and their complement is computed, and the higher value is chosen.')
        kmeans_accuracy = accuracy_score(y, clusters)

        st.markdown("##### Accuracy Score: {:.2f}%".format(kmeans_accuracy * 100))

        # --- Elbow Method + Silhouette Scores ---
        st.subheader('Evaluating the Best Number of Clusters')
        st.write('The process of determining the optimal number of clusters is a common challenge in clustering applications. Two popular methods to address this concern are:')
        st.markdown('**Elbow Method:** plot the Within-Cluster Sum of Squares (WCSS) against different values of k. The "elbow" point, where the rate of decrease sharply changes, suggests an optimal value for k.')
        st.markdown('**Silhouette Score:** quantifies how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better clustering. The average silhouette score is calculatedfor different values of k and the one with the highest score is selected.')
        ks = range(2, k + 1)
        wcss = []
        silhouette_scores = []
        
        for k in ks: # Loops over a range of k values; computes WCSS and silhouette score for each one to help identify the optimal number of clusters.
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X_std)
            wcss.append(km.inertia_)
            labels = km.labels_
            silhouette_scores.append(silhouette_score(X_std, labels))
        
        # A DataFrame summarizing the scores
        metrics_df = pd.DataFrame({
            'WCSS': wcss,
            'Silhouette Score': silhouette_scores
        }, index=ks)
        metrics_df.index.name = "k"  # Label the index for clarity

        # Display metrics in a DataFrame
        st.dataframe(metrics_df.style.format({"WCSS": "{:.2f}", "Silhouette Score": "{:.3f}"}))

        # --- Plot 3: Elbow Method and Silhouette Scores ---
        st.markdown('#### Visualizations for Elbow Method and Silhouette Score')
        st.write('Try changing the number of clusters (k) to see how these visuals change')
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

        # Elbow curve (WCSS vs. k)
        ax3a.plot(ks, wcss, marker='o')
        ax3a.set_xlabel('Number of Clusters (k)')
        ax3a.set_ylabel('WCSS')
        ax3a.set_title('Elbow Method for Optimal k')
        ax3a.grid(True)

        # Silhouette score vs. k
        ax3b.plot(ks, silhouette_scores, marker='o', color='green')
        ax3b.set_xlabel('Number of Clusters (k)')
        ax3b.set_ylabel('Silhouette Score')
        ax3b.set_title('Silhouette Score for Optimal k')
        ax3b.grid(True)

        plt.tight_layout()
        st.pyplot(fig3)

    if model_option == 'Hierarchical Clustering': # Only run this block if the user selected Hierarchical Clustering in the sidebar.

        st.subheader('Hierarchical Clustering')
        st.write('The unsupervised learning model you have chosen is: Hierarchical Clustering. Hierarchical Clustering is an unsupervised machine learning algorithm used to group similar data points into clusters based on their distance or similarity to one another.')
        st.write('Note: since Hierarchical Clustering relies on Euclidian distance, the data is centered and scaled')
        
        # Uses StandardScaler to center and scale the features: this is crucial for distance-based clustering methods like hierarchical clustering.
        scaler = StandardScaler() 
        X_scaled = scaler.fit_transform(X)

        # --- Sidebar Controls ---
        st.sidebar.subheader("Clustering Options")
        linkage_option = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"]) # Lets the user choose how distances between clusters are calculated.
        k = st.sidebar.slider('Select number of clusters (k)', min_value=2, max_value=10, value=2, step=1) # Lets the user choose how many clusters they want to form.

        # --- Dendrogram ---
        st.subheader('Dendrogram')
        st.write('The dendrogram, or the hierarchical tree, is a visual representation of the hierarchical clustering process: it shows how individual data points are grouped step by step into clusters, based on their similarity. The dendrogram provides two insights:  similarity structure (who merges early), and reasonable cut heights (horizontal line) for k clusters.')
        st.write('The linkage option in hierarchical clustering can be changed in the sidebar, which affects how distances between clusters are calculated, which in turn impacts how clusters are formed and what the dendrogram looks like. The linkage methods are as follows:')
        st.markdown('**Ward (default in many cases):** merges clusters that result in the smallest increase in total within-cluster variance.')
        st.markdown('**Single:** merges clusters with the smallest minimum distance between any two points.')
        st.markdown('**Complete:** merges clusters with the smallest maximum distance between points.')
        st.markdown('**Average:** uses the average distance between all points in two clusters.')
        st.write('Try changing the linkage method in the sidebar to see different results for both the dendrogram.')
        Z = linkage(X_scaled, method=linkage_option) # Computes the hierarchical clustering tree (Z) using the selected linkage method.

        if y is not None:
            labels = y.astype(str).tolist() # Sets labels for each sample in the dendrogram (true labels if available, or row indices).
        else:
            labels = df.index.astype(str).tolist()

        fig1, ax1 = plt.subplots(figsize=(20, 7))
        dendrogram(Z, truncate_mode='lastp', labels=labels, ax=ax1)
        ax1.set_title("Hierarchical Clustering Dendrogram")
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("Distance")
        st.markdown('#### Dendrogram (Hierarchical Tree) Visualization')
        st.pyplot(fig1)

        # --- Final Clustering with User-Selected k ---
        # Applies agglomerative clustering with the chosen k and linkage method.
        agg = AgglomerativeClustering(n_clusters=k, linkage=linkage_option)
        cluster_labels = agg.fit_predict(X_scaled)

        # Adds the cluster assignments to a copy of the original dataset.
        clustered_df = df.copy()
        clustered_df["Cluster"] = cluster_labels

        # --- Clustered Table Preview ---
        st.write("### Cluster Assignments (First 10 Rows)")
        st.dataframe(clustered_df[[target_column, "Cluster"]].head(10)) # Shows the first 10 rows of the data with cluster assignments.

        st.write("### Cluster Sizes")
        st.dataframe(clustered_df["Cluster"].value_counts().reset_index().rename(columns={"index": "Cluster", "Cluster": "Count"})) # Displays the size (number of rows) in each cluster.

        # --- PCA Visualization ---
        st.subheader('Low‑Dimensional Insight with PCA')
        st.markdown('PCA is **only** for display; it is **not** used to fit the clusters.')
        # Reduces the data to 2D for visualization.
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.markdown("##### Cluster Visualization (PCA Reduced)") # Displays a colored scatter plot of the clusters in PCA space.
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        # Uses color to differentiate clusters and adds a legend.
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=60, edgecolor='k', alpha=0.7)
        ax2.set_xlabel("Principal Component 1")
        ax2.set_ylabel("Principal Component 2")
        ax2.set_title(f"Agglomerative Clustering (PCA View, k={k}, linkage='{linkage_option}')")
        ax2.legend(*scatter.legend_elements(), title="Clusters")
        ax2.grid(True)
        st.pyplot(fig2)

        # --- Silhouette Score Curve ---
        st.subheader("Silhouette Score Analysis")
        st.markdown('**Silhouette Score Analysis** is a method used to evaluate the quality of clustering in unsupervised machine learning. It tells you how well each data point fits within its assigned cluster and how distinct the clusters are from one another.')
        st.write('The average silhouette score for all data points gives an overall measure of clustering performance.')

        # Set up a loop range and an empty list to compute silhouette scores for different cluster sizes
        k_range = range(2, 11)
        sil_scores = []

        for k_test in k_range: # Computes silhouette scores for k ranging from 2 to 10 to assess cluster quality.
            temp_labels = AgglomerativeClustering(n_clusters=k_test, linkage=linkage_option).fit_predict(X_scaled)
            score = silhouette_score(X_scaled, temp_labels)
            sil_scores.append(score)

        best_k = k_range[np.argmax(sil_scores)] # Identifies and displays the k with the best silhouette score

        fig_sil, ax_sil = plt.subplots(figsize=(7, 4))
        ax_sil.plot(list(k_range), sil_scores, marker="o")
        ax_sil.set_xticks(list(k_range))
        ax_sil.set_xlabel("Number of Clusters (k)")
        ax_sil.set_ylabel("Average Silhouette Score")
        ax_sil.set_title("Silhouette Analysis for Agglomerative Clustering")
        ax_sil.grid(True, alpha=0.3)
        st.pyplot(fig_sil)

        st.write(f"**Best k by silhouette score: {best_k}**  _(score = {max(sil_scores):.3f})_")