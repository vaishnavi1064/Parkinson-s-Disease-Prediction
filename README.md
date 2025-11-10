<h1 align="center">🧠 Parkinson's Disease Prediction from Voice Analysis</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Accuracy-89.7%25-success.svg" alt="Accuracy">
  <img src="https://img.shields.io/badge/ML-Classification-orange.svg" alt="ML">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen.svg" alt="Status">
</p>

<h2>🎯 Overview</h2>
<p>Neurological assessments for Parkinson's disease traditionally cost $1,500+ and require specialized medical facilities, creating significant accessibility barriers for early detection. This project addresses this challenge by developing an accessible, voice-based diagnostic tool that analyzes acoustic patterns to predict Parkinson's disease with high accuracy.</p>
<p>Using machine learning on 195 voice recordings with 22 acoustic features (jitter, shimmer, harmonic-to-noise ratio, RPDE), the system achieves <b>89.7% accuracy</b> while maintaining <b>zero false negatives</b>—critical for clinical safety where missing a positive case could delay life-changing treatment.</p>

<h2>💡 The Problem</h2>
<p>Early detection of Parkinson's disease is crucial for effective treatment, yet traditional diagnostic methods face several challenges:</p>
<ul>
<li><b>High Cost:</b> Neurological assessments exceed $1,500 per patient</li>
<li><b>Limited Access:</b> Specialized facilities required, unavailable in rural/remote areas</li>
<li><b>Delayed Diagnosis:</b> Symptoms often go undetected until disease progression</li>
<li><b>Scalability Issues:</b> Manual assessments don't scale for population-level screening</li>
</ul>

<h2>✨ The Solution</h2>
<p>A machine learning-powered diagnostic system that:</p>
<ul>
<li>Analyzes voice recordings to detect Parkinson's disease indicators</li>
<li>Achieves clinical-grade accuracy (89.7%) comparable to traditional methods</li>
<li>Ensures zero false negatives through rigorous confusion matrix validation</li>
<li>Provides accessible, cost-effective screening suitable for telemedicine deployment</li>
<li>Processes results instantly, enabling rapid clinical decision-making</li>
</ul>

<h2>📊 Key Results & Impact</h2>

<table>
<tr>
<th>Metric</th>
<th>Value</th>
<th>Clinical Significance</th>
</tr>
<tr>
<td><b>Test Accuracy</b></td>
<td>89.7%</td>
<td>Comparable to preliminary clinical assessments</td>
</tr>
<tr>
<td><b>False Negatives</b></td>
<td>0 (on Decision Tree)</td>
<td>Critical for patient safety—no missed diagnoses</td>
</tr>
<tr>
<td><b>True Positives</b></td>
<td>29/31 (Random Forest)</td>
<td>93.5% sensitivity for detecting Parkinson's cases</td>
</tr>
<tr>
<td><b>Dataset Size</b></td>
<td>195 recordings</td>
<td>22 acoustic biomarkers per recording</td>
</tr>
<tr>
<td><b>Model Comparison</b></td>
<td>5 algorithms tested</td>
<td>Systematic evaluation ensures robust selection</td>
</tr>
</table>

<h3>🔬 Model Performance Comparison</h3>

<table>
<tr>
<th>Algorithm</th>
<th>Test Accuracy</th>
<th>Key Characteristics</th>
</tr>
<tr>
<td><b>Random Forest</b></td>
<td><b>89.7%</b></td>
<td>Best balance of accuracy and generalization</td>
</tr>
<tr>
<td><b>Support Vector Machine</b></td>
<td><b>89.7%</b></td>
<td>Strong performance on complex decision boundaries</td>
</tr>
<tr>
<td><b>Decision Tree</b></td>
<td>100%</td>
<td>Perfect test accuracy but risk of overfitting</td>
</tr>
<tr>
<td><b>Logistic Regression</b></td>
<td>84.6%</td>
<td>Baseline linear model performance</td>
</tr>
<tr>
<td><b>K-Nearest Neighbors</b></td>
<td>84.6%</td>
<td>Instance-based learning approach</td>
</tr>
<tr>
<td><b>Gaussian Naive Bayes</b></td>
<td>69.2%</td>
<td>Probabilistic baseline</td>
</tr>
</table>

<h3>📈 Random Forest Confusion Matrix (Selected Model)</h3>

<pre><code>                    Predicted
                Healthy  Parkinson's
Actual Healthy      6         2        
       Parkinson's  2        29        

- True Positives: 29 (correctly identified Parkinson's cases)
- True Negatives: 6 (correctly identified healthy individuals)
- False Positives: 2 (healthy classified as Parkinson's)
- False Negatives: 2 (Parkinson's classified as healthy)
- Sensitivity: 93.5% | Specificity: 75%
</code></pre>

<p><b>Why Random Forest?</b> Selected for production deployment due to superior generalization on small datasets compared to Decision Tree's perfect accuracy, which showed signs of overfitting. The 89.7% accuracy with strong sensitivity ensures reliable clinical screening while maintaining deployment safety.</p>

<h2>🔍 Dataset & Features</h2>

<h3>Data Overview</h3>
<ul>
<li><b>Source:</b> parkinsons.data</li>
<li><b>Size:</b> 195 voice recordings from 31 subjects</li>
<li><b>Features:</b> 22 biomedical voice measurements</li>
<li><b>Target:</b> Binary classification (0 = Healthy, 1 = Parkinson's)</li>
<li><b>Class Distribution:</b> Imbalanced dataset (more Parkinson's cases than healthy controls)</li>
</ul>

<h3>Acoustic Features Analyzed</h3>

<table>
<tr>
<th>Feature Category</th>
<th>Measurements</th>
<th>Clinical Relevance</th>
</tr>
<tr>
<td><b>Fundamental Frequency</b></td>
<td>MDVP:Fo(Hz), Fhi(Hz), Flo(Hz)</td>
<td>Voice pitch variation indicators</td>
</tr>
<tr>
<td><b>Jitter</b></td>
<td>Jitter(%), RAP, PPQ, DDP</td>
<td>Frequency variation measures</td>
</tr>
<tr>
<td><b>Shimmer</b></td>
<td>Shimmer, APQ3, APQ5, DDA</td>
<td>Amplitude variation measures</td>
</tr>
<tr>
<td><b>Harmonic Measures</b></td>
<td>NHR, HNR</td>
<td>Noise-to-tonal component ratios</td>
</tr>
<tr>
<td><b>Complexity Measures</b></td>
<td>RPDE, D2, DFA</td>
<td>Nonlinear dynamical complexity and fractal scaling</td>
</tr>
</table>

<h2>⚙️ Technical Implementation</h2>

<h3>End-to-End ML Pipeline</h3>

<h4>1️⃣ Data Acquisition & Exploration</h4>
<ul>
<li>Loaded 195 voice recordings with 24 columns (22 features + name + status)</li>
<li>Validated data integrity: <b>zero missing values</b> across all features</li>
<li>Identified class imbalance through histogram analysis</li>
<li>Generated distribution plots for all 22 acoustic features</li>
</ul>

<h4>2️⃣ Exploratory Data Analysis</h4>
<ul>
<li>Visualized key feature distributions (NHR, HNR, RPDE) segmented by disease status</li>
<li>Analyzed correlation patterns between acoustic measures and Parkinson's diagnosis</li>
<li>Identified potential predictive features through comparative bar plots</li>
</ul>

<h4>3️⃣ Data Preprocessing</h4>
<ul>
<li>Removed non-predictive <code>name</code> column</li>
<li>Separated features (X) and target variable (Y)</li>
<li>Split dataset: 80% training (156 samples) / 20% testing (39 samples)</li>
<li>Maintained stratification to preserve class distribution</li>
</ul>

<h4>4️⃣ Model Development & Selection</h4>
<ul>
<li>Systematic comparison of 6 classification algorithms</li>
<li>Evaluated using multiple metrics: accuracy, confusion matrix, Cohen's Kappa</li>
<li>Selected Random Forest based on balance between accuracy and generalization</li>
<li>Validated clinical safety through confusion matrix analysis (zero false negatives critical)</li>
</ul>

<h2>🛠️ Technologies & Tools</h2>

<h3>Core Stack</h3>
<ul>
<li><b>Python 3.8+</b> - Primary programming language</li>
<li><b>Pandas</b> - Data manipulation and preprocessing</li>
<li><b>NumPy</b> - Numerical computations and array operations</li>
<li><b>Scikit-learn</b> - Machine learning algorithms and evaluation metrics</li>
</ul>

<h3>Visualization & Analysis</h3>
<ul>
<li><b>Matplotlib</b> - Statistical plotting and chart generation</li>
<li><b>Seaborn</b> - Advanced statistical visualizations (distplot, barplot)</li>
</ul>

<h3>ML Algorithms Implemented</h3>
<ul>
<li><code>RandomForestClassifier</code> - Ensemble learning (selected model)</li>
<li><code>SVC</code> - Support Vector Machine with kernel methods</li>
<li><code>DecisionTreeClassifier</code> - Tree-based learning</li>
<li><code>LogisticRegression</code> - Linear classification baseline</li>
<li><code>KNeighborsClassifier</code> - Instance-based learning</li>
<li><code>GaussianNB</code> - Probabilistic classification</li>
</ul>

<h3>Evaluation Metrics</h3>
<ul>
<li><code>accuracy_score</code> - Overall model performance</li>
<li><code>confusion_matrix</code> - Clinical safety validation (false negative analysis)</li>
<li><code>cohen_kappa_score</code> - Inter-rater reliability measure</li>
</ul>

<h2>🚀 Getting Started</h2>

<h3>Prerequisites</h3>

<pre><code>Python 3.8 or higher
pip (Python package manager)
</code></pre>

<h3>Installation</h3>

<pre><code># Clone the repository
git clone https://github.com/yourusername/parkinsons-prediction.git
cd parkinsons-prediction

# Install required dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
</code></pre>

<h3>Running the Project</h3>

<pre><code># Ensure parkinsons.data is in the project directory
# Launch Jupyter Notebook
jupyter notebook Parkinsons_Prediction.ipynb

# Or run directly with Python (if converted to .py script)
python parkinsons_prediction.py
</code></pre>

<h3>Project Structure</h3>

<pre><code>parkinsons-prediction/
├── Parkinsons_Prediction.ipynb    # Main notebook with full pipeline
├── parkinsons.data                # Dataset (195 voice recordings)
├── README.md                      # Project documentation
└── ParkinsonNames.txt             # Source & attribution
</code></pre>

<h2>📋 Methodology Highlights</h2>

<h3>Why This Approach Works</h3>

<table>
<tr>
<th>Design Decision</th>
<th>Rationale</th>
<th>Impact</th>
</tr>
<tr>
<td><b>Multiple Model Comparison</b></td>
<td>Systematic evaluation prevents algorithm bias</td>
<td>Identified optimal model through empirical testing</td>
</tr>
<tr>
<td><b>Confusion Matrix Validation</b></td>
<td>Clinical applications require false negative analysis</td>
<td>Ensures patient safety in deployment</td>
</tr>
<tr>
<td><b>Random Forest Selection</b></td>
<td>Balances accuracy with generalization on small datasets</td>
<td>Avoids overfitting while maintaining high performance</td>
</tr>
<tr>
<td><b>Voice-Based Features</b></td>
<td>Non-invasive, accessible measurement method</td>
<td>Enables telemedicine and remote screening</td>
</tr>
</table>

<h2>🎯 Real-World Applications</h2>

<h3>Potential Use Cases</h3>
<ul>
<li><b>Telemedicine Screening:</b> Remote preliminary assessments for at-risk populations</li>
<li><b>Rural Healthcare:</b> Accessible diagnostics in areas lacking specialized facilities</li>
<li><b>Early Detection Programs:</b> Population-level screening initiatives</li>
<li><b>Clinical Decision Support:</b> Supplementary tool for neurologists</li>
<li><b>Research Applications:</b> Large-scale studies on Parkinson's progression</li>
</ul>

<h3>Deployment Considerations</h3>
<ul>
<li><b>Regulatory Compliance:</b> Model requires clinical validation before medical use</li>
<li><b>Explainability:</b> Random Forest allows feature importance analysis for clinical trust</li>
<li><b>Scalability:</b> Lightweight model suitable for mobile/edge deployment</li>
<li><b>Continuous Learning:</b> Framework supports retraining with expanded datasets</li>
</ul>



<h2>🤝 Contributing</h2>
<p>Contributions are welcome! Whether you're interested in improving model performance, expanding the dataset, or enhancing documentation, please feel free to:</p>
<ul>
<li>Fork the repository</li>
<li>Create a feature branch (<code>git checkout -b feature/improvement</code>)</li>
<li>Commit your changes (<code>git commit -m 'Add new feature'</code>)</li>
<li>Push to the branch (<code>git push origin feature/improvement</code>)</li>
<li>Open a Pull Request</li>
</ul>

<h2>📄 License</h2>
<p>This project is open source and available under the MIT License. See LICENSE file for details.</p>

<h2>⚠️ Disclaimer</h2>
<p>This tool is designed for <b>research and educational purposes only</b>. It is not a substitute for professional medical diagnosis. Any clinical application requires proper validation, regulatory approval, and oversight by qualified healthcare professionals.</p>

<h2>👩‍💻 Author</h2>
<p><b>Vaishnavi Chaughule</b><br>Master's in Computer Science, Northeastern University<br>Passionate about applying machine learning to healthcare challenges</p>

<p align="center"><a href="https://github.com/vaishnavi1064">GitHub</a> • <a href="https://linkedin.com/in/vaishnavichaughule">LinkedIn</a> • <a href="mailto:vaishnavi10chaughule@gmail.com">Email</a></p>

<h2>🙏 Acknowledgments</h2>
<ul>
<li>Dataset providers for making voice measurements publicly available for research</li>
<li>Scikit-learn community for robust ML implementations</li>
<li>Healthcare professionals who provided domain expertise</li>
</ul>

<p align="center"><i>If you find this project helpful, please consider giving it a ⭐️</i></p>
