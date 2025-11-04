<h1 align="center">🧠 <b>Parkinson's Disease Prediction from Voice Measures</b></h1>

<h2>🎯 <b>Project Overview</b></h2>
<p>
This project uses machine learning to predict whether a patient is suffering from Parkinson's disease based on a range of vocal measurements. The goal is to build a classification model that can accurately distinguish between a healthy individual (status 0) and an individual with Parkinson's (status 1).
</p>
<p>
The notebook walks through the full data science pipeline: data loading, exploratory data analysis (EDA), visualization, preprocessing, and model building, and comparison.
</p>

<h2>🧾 <b>Dataset</b></h2>
<p>
The dataset used is <b>"parkinsons.data"</b>, which contains 195 rows and 24 columns.
</p>
<ul>
<li><b>Features:</b> 22 biomedical voice measures, including:
<ul>
<li>Average vocal fundamental frequency (MDVP:Fo(Hz))</li>
<li>Maximum and minimum vocal fundamental frequency (MDVP:Fhi(Hz), MDVP:Flo(Hz))</li>
<li>Several measures of variation in fundamental frequency (Jitter)</li>
<li>Several measures of variation in amplitude (Shimmer)</li>
<li>Ratio of noise to tonal components (NHR, HNR)</li>
<li>Nonlinear dynamical complexity measures (RPDE, D2)</li>
<li>Signal fractal scaling exponent (DFA)</li>
</ul>
</li>
<li><b>Target Variable:</b> <code>status</code> - (1) for Parkinson's, (0) for healthy.</li>
</ul>

<h2>⚙️ <b>Project Workflow</b></h2>

<h3>1️⃣ Data Loading & EDA</h3>
<ul>
<li>Loaded the <code>parkinsons.data</code> file into a pandas DataFrame.</li>
<li>Checked the data structure with <code>.info()</code>, <code>.shape</code>, and <code>.describe()</code>.</li>
<li>Confirmed that the dataset has <b>no missing (null) values</b>.</li>
</ul>

<h3>2️⃣ Visualization</h3>
<ul>
<li>Plotted a histogram of the <code>status</code> column to visualize the class distribution, noting an imbalance (more patients with Parkinson's than healthy controls).</li>
<li>Used <code>seaborn.barplot</code> to compare key features like <b>NHR</b>, <b>HNR</b>, and <b>RPDE</b> against the <code>status</code>.</li>
<li>Generated <code>seaborn.distplot</code> for all features to observe their data distribution.</li>
</ul>

<h3>3️⃣ Preprocessing & Splitting</h3>
<ul>
<li>Dropped the non-numeric <code>name</code> column as it is not a predictive feature.</li>
<li>Separated the data into features (<b>X</b>) and the target variable (<b>Y</b>).</li>
<li>Split the data into training and testing sets (80% train, 20% test) using <code>train_test_split</code>.</li>
</ul>

<h3>4️⃣ Model Building & Evaluation</h3>
<ul>
<li>Trained and evaluated several classification models to find the best performer.</li>
<li>Models were evaluated using <b>Accuracy Score</b>, <b>Confusion Matrix</b>, and <b>Cohen's Kappa Score</b>.</li>
</ul>

<h2>🤖 <b>Models Compared</b></h2>
<ul>
<li><b>Logistic Regression:</b> 84.6% Test Accuracy</li>
<li><b>Random Forest Classifier:</b> 89.7% Test Accuracy</li>
<li><b>Decision Tree Classifier:</b> 100% Test Accuracy (Note: This shows signs of overfitting or data leakage from the notebook).</li>
<li><b>Gaussian Naive Bayes:</b> 69.2% Test Accuracy</li>
<li><b>K-Nearest Neighbors (KNN):</b> 84.6% Test Accuracy</li>
<li><b>Support Vector Machine (SVC):</b> 89.7% Test Accuracy</li>
</ul>

<h2>📊 <b>Results</b></h2>
<p>
The <b>Random Forest Classifier</b> and <b>Support Vector Machine (SVC)</b> provided the best and most reliable performance on the unseen test data, both achieving an accuracy of <b>89.7%</b>.
</p>
<p>The confusion matrix for the Random Forest model was:</p>
<pre><code>
Test Confusion Matrix:
[[ 6  2]
[ 2 29]]

True Positives (Parkinson's): 29

True Negatives (Healthy): 6

False Positives: 2

False Negatives: 2
</code></pre>

<h2>🧰 <b>Technologies & Libraries Used</b></h2>
<ul>
<li><b>Python 3</b></li>
<li><b>Pandas</b> (for data manipulation)</li>
<li><b>NumPy</b> (for numerical operations)</li>
<li><b>Matplotlib</b> & <b>Seaborn</b> (for data visualization)</li>
<li><b>Scikit-learn</b> (for machine learning), including:
<ul>
<li><code>LogisticRegression</code></li>
<li><code>RandomForestClassifier</code></li>
<li><code>DecisionTreeClassifier</code></li>
<li><code>GaussianNB</code></li>
<li><code>KNeighborsClassifier</code></li>
<li><code>SVC</code></li>
<li><code>train_test_split</code></li>
<li><code>accuracy_score</code>, <code>confusion_matrix</code>, <code>cohen_kappa_score</code></li>
</ul>
</li>
<li><b>Jupyter Notebook</b> (as the IDE)</li>
</ul>

<h2>🚀 <b>How to Run</b></h2>
<ol>
<li>Clone the repository.</li>
<li>Ensure you have Python and the libraries listed above installed (<code>pip install pandas numpy matplotlib seaborn scikit-learn</code>).</li>
<li>Place the <code>parkinsons.data</code> file in the correct directory.</li>
<li>Run the Jupyter Notebook (<code>Parkinsons_Prediction.ipynb</code>) using Jupyter Lab or Jupyter Notebook.</li>
</ol>

<h2>👩‍💻 <b>Author</b></h2>
<p>
<b>Vaishnavi Chaughule</b>
</p>
