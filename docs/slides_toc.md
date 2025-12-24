### **BA1: Introduction to Business Data Analytics**
**Focus:** CRISP-DM, KPI Trees, and Problem Framing

*   **Introduction & Context**
    *   **Course Agenda:** Overview of the 3-hour structure covering Intro, CRISP-DM, KPI Trees, and Problem Framing.
    *   **Learning Outcomes:** Understanding the analytics ecosystem, applying CRISP-DM, designing KPI trees, and framing business problems.
    *   **Definition:** Defining Business Data Analytics as a process turning data into insight/action using descriptive, diagnostic, predictive, and prescriptive layers.
    *   **BI vs. DS/AI:** Distinguishing between Business Intelligence (past/monitoring) and Data Science/AI (future/automation).
    *   **Data Value Chain:** The flow from Collection to Monitoring, emphasizing product thinking.
    *   **Team Roles & Skills:** Breakdown of roles (PM, Analyst, Scientist, Engineer) and necessary skills (SQL, Stats, EDA).
    *   **Ethics & Governance:** Privacy, bias, explainability, and data governance concepts (lineage, RACI).
    *   **Class Case Study:** Increasing E-commerce revenue by 15% using specific levers (CR, AOV, Retention).

*   **CRISP-DM Framework**
    *   **Overview:** The six phases (Business Understanding to Deployment) and iterative nature.
    *   **Phase 1 (Business Understanding):** Clarifying objectives, identifying decisions, and defining success KPIs (using the E-commerce example),.
    *   **Phase 2 (Data Understanding):** Inventorying sources, EDA, reconciling KPI definitions, and "quick wins" like funnel analysis,.
    *   **Phase 3 (Data Preparation):** Cleaning, feature engineering (RFM), and data quality tests,.
    *   **Phase 4 (Modeling):** Selecting tasks (classification/regression), baselines, and the improvement loop.
    *   **Phase 5 (Evaluation):** Technical metrics (AUC/RMSE) vs. Business metrics (Profit/Lift),.
    *   **Phase 6 (Deployment):** Strategies (Batch vs. Real-time), monitoring for drift, and MLOps.
    *   **Handover & Pitfalls:** Deliverables (Project Charter) and common failures (no adoption, leakage),.
    *   **Exercise 1:** Drafting a project charter and RACI for the class case.

*   **KPI Trees**
    *   **Definitions:** Leading vs. Lagging indicators and the principles of good metrics.
    *   **5-Step Process:** From North Star to drivers, levers, guardrails, and dashboards.
    *   **Example (E-commerce):** Decomposing Revenue into Sessions, CR, and AOV with specific numeric illustrations.
    *   **Example (SaaS):** Decomposing MRR into Customers, ARPA, and Churn.
    *   **Measurement & Guardrails:** Establishing canonical definitions and monitoring health metrics (latency, errors).
    *   **Pitfalls:** Overlapping branches, missing guardrails, and slow refresh rates.
    *   **Exercise 2:** Building a KPI tree for a ride-hailing app.

*   **Problem Framing**
    *   **Importance:** Why framing prevents analytics failure.
    *   **DOC Template:** Context, Decision, Options, and Criteria.
    *   **Translation:** Converting business questions into specific analytics tasks (diagnostic vs. prescriptive).
    *   **Target Variables:** Defining targets while avoiding leakage,.
    *   **Constraints & Risks:** Identifying resource limits, assumptions, and external risks.
    *   **Success Criteria:** Defining primary success metrics and necessary guardrails (technical/product/ethical).
    *   **Prioritization:** Impact/Effort matrix for ranking opportunities.
    *   **A/B Testing Basics:** Randomization units and sample size rules of thumb.
    *   **Exercise 3:** Writing a problem statement using the DOC framework.
    *   **Anti-patterns:** Common mistakes like vague KPIs or lack of decision alignment.

*   **Conclusion**
    *   **Key Takeaways:** Summary of the four main sections.
    *   **References:** Suggested reading (Davenport, Provost).
    *   **Homework:** Building a KPI tree and problem statement for a digital product.

---

### **BA2: Data Architecture & Advanced SQL**
**Focus:** Architecture Patterns, Joins, Window Functions, and Pipeline Design

*   **Data Architecture**
    *   **Agenda:** Architecture patterns, Advanced Joins, CTEs, Window Functions, and Time Travel.
    *   **OLTP vs. OLAP:** Comparing transactional (write-heavy) vs. analytical (read-heavy) workloads.
    *   **Dimensional Modeling:** Star Schema (Fact/Dimensions) vs. Snowflake Schema (normalized dimensions),.
    *   **SCD Type 2:** Handling slowly changing dimensions with validity ranges and SQL implementation,.
    *   **Lakehouse & Medallion:** Bronze (Raw), Silver (Cleaned), and Gold (Curated) architecture.
    *   **Batch vs. Streaming:** Trade-offs and CDC (Change Data Capture) concepts.
    *   **Data Model:** The schema for the hands-on event pipeline (orders, payments, sessions).

*   **Advanced SQL Techniques**
    *   **Joins:** Equi vs. Non-Equi joins and Inner vs. Outer logic.
    *   **Semi & Anti Joins:** Using `EXISTS` and `NOT EXISTS` for filtering without multiplying rows.
    *   **Gotchas:** Handling NULLs in equality checks and avoiding row explosion.
    *   **Performance:** Broadcast vs. Shuffle joins and optimization strategies.
    *   **CTEs:** Using Common Table Expressions for readability and organizing complex logic.
    *   **Recursive CTEs:** Syntax and use cases for tree traversal and hierarchies,.

*   **Window Functions**
    *   **Syntax:** Partitioning, ordering, and framing (ROWS vs. RANGE).
    *   **Rolling Metrics:** Calculating 7-day rolling revenue using window frames.
    *   **Ranking:** Using `ROW_NUMBER`, `RANK` for "Top-N" analysis.
    *   **Sessionization:** Using `LAG()` to define sessions based on time gaps.
    *   **Cohorts:** Calculating retention and LTV using first-value windows.

*   **Time Travel & Pipelines**
    *   **Concept:** Querying historical data states for auditing or recovery.
    *   **Syntax:** Examples for Snowflake (`AT`), BigQuery (`FOR SYSTEM_TIME`), and Delta Lake (`VERSION AS OF`),.
    *   **Hands-on Pipeline:**
        *   **Bronze Layer:** Raw ingestion and validation.
        *   **Silver Layer:** Conforming data and standardizing status values.
        *   **Gold Layer:** Creating core data marts (Daily Revenue, AOV).
        *   **Funnel Analysis:** Linking sessions to orders to payments.
        *   **Data Quality Checks:** Detecting payment mismatches and anomalies,.
        *   **Rolling KPIs:** Window functions for trends.
        *   **User Cohorts:** Building retention tables.
        *   **End-to-End Example:** A single complex query demonstrating the full flow.

*   **Wrap-up**
    *   **Exercises:** Tasks for computing conversion rates, finding anomalies, and recursive CTEs.
    *   **References:** Kimball & Ross, official documentation.

---

### **BA3: Data Quality & Feature Engineering**
**Focus:** Data Profiling, Missingness, Leakage, Encoding, and Pipelines

*   **Introduction & Data Quality**
    *   **Agenda & Case:** Overview and introduction to the Telco Churn prediction case study,.
    *   **Dimensions:** Accuracy, Completeness, Consistency, Timeliness, and Validity.
    *   **Scorecard & Lifecycle:** Defining KPIs for quality and the lifecycle from ingest to monitoring.
    *   **Profiling:** Using distributions and visuals to understand data health.
    *   **Specific Issues:** Duplicates, orphan records, and unit/domain validity.

*   **Missing Data**
    *   **Mechanisms:** MCAR (Random), MAR (Dependent on observed data), MNAR (Dependent on missing value).
    *   **Diagnosis:** Using missingness maps and segment comparisons,.
    *   **Strategies:** Dropping rows/cols, simple imputation (mean/median), and flagging missingness.
    *   **Advanced Imputation:** kNN imputation and MICE (Multiple Imputation by Chained Equations).
    *   **Business Rules:** Time-aware and domain-specific logic for imputation.
    *   **Assessment:** Comparing model metrics pre- and post-imputation.
    *   **Pitfalls:** Leakage during imputation and forgetting flags.

*   **Data Leakage**
    *   **Definition:** Information from the future contaminating training data.
    *   **Types:** Target leakage, Train-test contamination, and Temporal leakage.
    *   **Prevention:** Using time-based splits and "Anti-Leakage" pipeline principles (fit on train only).

*   **Feature Engineering & Selection**
    *   **Categorical Encoding:** One-Hot (pros/cons), Ordinal, and Target/Mean encoding (with CV safety)-.
    *   **Advanced Encoding:** Weight of Evidence (WoE), Hashing Trick, and handling high cardinality.
    *   **Text & Embeddings:** Bag-of-Words, TF-IDF, and Entity Embeddings,.
    *   **Scaling:** When to use Standard, MinMax, or Robust scalers.
    *   **Transforms:** Log, Box-Cox, and Quantile transforms for normality.
    *   **Outliers:** Detection (Z-score, IQR, Isolation Forest) and handling (capping, removal).
    *   **Feature Types:** Date-time, Geo (Haversine), Aggregations, Time-series lags, and NLP sentiment,.
    *   **Selection Methods:** Filter (Chi-square), Wrapper (RFE), Embedded (Lasso, Tree importance), and Dimensionality Reduction (PCA)-.

*   **Pipelines & Workflow**
    *   **Golden Rules:** Never fit on full data; use ColumnTransformers.
    *   **Implementation:** Pseudocode for a leakage-safe Scikit-Learn pipeline-.
    *   **Evaluation:** Comparing metrics after preprocessing.
    *   **Validation:** Schema contracts, pre-deployment checks, and monitoring.
    *   **Mini-Case:** Three-part exercise on profiling, imputation/encoding, and modeling,.

*   **Conclusion**
    *   **Takeaways:** Quality sets the ceiling; pipelines are mandatory.
    *   **Appendix:** Formulas for Z-score, IQR, and code snippets for Target Encoding,.

---

### **BA4: EDA & Visualization for Decision Making**
**Focus:** Exploratory Analysis, Visualization Principles, and Causal Reasoning

*   **Fundamentals**
    *   **Agenda:** Fundamentals, Univariate/Bivariate/Multivariate analysis, Correlation vs. Causation.
    *   **Telco Case:** Introduction to the 1,200 customer synthetic dataset.
    *   **Mindset:** Iterative profiling, hypothesis testing, and triangulation.
    *   **Data Types:** Numeric vs. Categorical and Date-time handling.

*   **Analysis Levels**
    *   **Univariate:** Goals (shape, center, spread), Histograms (binning), Boxplots, and Summary Stats-.
    *   **Bivariate:**
        *   Numeric-Numeric: Scatter plots and correlation.
        *   Numeric-Categorical: Grouped boxplots.
        *   Categorical-Categorical: Contingency tables and Chi-square.
    *   **Multivariate:** Revealing joint structures, segmentation, and small multiples,.
    *   **Dimensionality Reduction:** Using PCA/t-SNE for visualization.
    *   **Interactions:** Creating ratios and composite features.

*   **Causality & Pitfalls**
    *   **Correlation vs. Causation:** The importance of reasoning and detecting spurious correlations.
    *   **Confounding:** Understanding confounders and colliders.
    *   **Simpson's Paradox:** How pooled trends can contradict within-group trends.
    *   **Testing:** Distinguishing between Observational studies and Randomized/Quasi-experiments.

*   **Decision Making & Design**
    *   **A/B Testing:** Practical definitions of metrics, randomization, and sample size.
    *   **From EDA to Decisions:** Moving from patterns to testable hypotheses and quantifying uncertainty.
    *   **Visualization Principles:** Marks & Channels, avoiding Chartjunk, and choosing the right chart type-.
    *   **Scales & Axes:** Best practices for baselines and binning.
    *   **Dashboards:** Designing for decisions and highlighting thresholds.
    *   **Workflow:** Reproducibility via scripts and versioning.

*   **Wrap-up**
    *   **Checklist:** A summary of steps for effective EDA.
    *   **Hands-on:** Tasks to recreate visuals and write a summary.
    *   **Quiz:** Questions on histograms, Simpson's paradox, and correlation.
    *   **References:** Anscombe, Tufte, and Scikit-learn docs.

---

### **BA5: Experimentation & Basic Causal Inference**
**Focus:** A/B Testing, Hypothesis Testing, Power Analysis, and Variance Reduction (CUPED)

*   **Introduction & Fundamentals**
    *   **Title Slide:** Lecture title and instructor information.
    *   **Learning Objectives:** Designing trustworthy A/B tests, understanding Type I/II errors, computing sample sizes, applying CUPED, and recognizing causal assumptions.
    *   **Agenda:** Experimentation fundamentals, Hypothesis testing, Power/MDE, CUPED, Multiple testing, and Causal thinking.
    *   **Why Experiment?:** Measuring causal impact, reducing decision risk, and building organizational learning.
    *   **Potential Outcomes:** Counterfactuals ($Y(1)$ vs $Y(0)$), ATE (Average Treatment Effect), and the ignorability assumption via random assignment.
    *   **Randomization:** Simple vs. stratified/block randomization and avoiding selection bias.
    *   **Experiment Lifecycle:** Define, Design, Run, Analyze, and Learn phases.
    *   **Metric Design:** Primary metrics (KPIs), Secondary metrics (diagnostics), and Guardrails (latency, errors).
    *   **Sanity Checks:** A/A tests, checking event counts, and detecting sample ratio mismatches (SRM).
    *   **SRM (Sample Ratio Mismatch):** Definition, causes (tracking bugs, bots), and the imperative to stop and investigate.

*   **Hypothesis Testing & Power**
    *   **Hypotheses & Errors:** Null vs. Alternative hypotheses, Type I (False Positive) vs. Type II (False Negative) errors.
    *   **p-values & Confidence Intervals:** Intuition behind p-values (data extremeness, not effect probability) and using CIs for plausible effects.
    *   **Two-Sample Tests:** Proportions z-test (conversion) and Welch’s t-test (means/revenue).
    *   **Effect Size:** Absolute vs. relative lift and defining practically significant MDE (Minimum Detectable Effect).
    *   **Power & Sample Size:** The relationship between Baseline $p_0$, MDE, $\alpha$, and desired power; trade-offs between MDE and sample size.
    *   **Sequential Testing:** The dangers of naïve peeking (inflating Type I error) and solutions like group-sequential methods.
    *   **Multiple Testing:** Controlling Family-Wise Error Rate (FWER) or False Discovery Rate (FDR) when using multiple metrics.
    *   **Distributional Issues:** Dealing with heavy tails (winsorization) and ratio metrics (Delta method).

*   **Variance Reduction (CUPED)**
    *   **Why Reduce Variance:** Lower noise leads to smaller required sample sizes or shorter tests.
    *   **CUPED Concept:** Using a pre-experiment covariate $X$ to adjust outcome $Y$.
    *   **Estimation:** Calculating $\theta$ using covariance and variance; estimating on pre-assignment data to avoid leakage.
    *   **Visualization:** Scatter plot illustrating the relationship between outcome and pre-experiment covariate.
    *   **Formula:** The adjusted metric $Y^* = Y - \theta(X - E[X])$.
    *   **Variance Reduction Factor:** How correlation $\rho$ impacts effective sample size gain ($\approx 1/(1-\rho^2)$).
    *   **Assumptions:** Linearity, proper randomization, and avoiding post-treatment covariates.
    *   **Recipe:** Step-by-step guide to implementing CUPED.
    *   **Synthetic Example:** Setup with Control/Treatment $N=5000$ and a known lift.
    *   **Results Table:** Comparing Effect Estimate and Standard Error for Unadjusted vs. CUPED.
    *   **Standard Error Chart:** Visual comparison of error reduction.
    *   **Implementation:** Pseudo-code for calculating theta and adjusted Y.

*   **Causal Thinking & Conclusion**
    *   **P-values Visuals:** Histograms of p-values under H0 (Uniform) vs. H1 (Skewed to 0).
    *   **Causal Thinking Basics:** How randomization breaks confounding; blocking and cluster randomization.
    *   **Reporting Results:** Stating effects with CIs, power achieved, and decision recommendations.
    *   **Mini-Case (Email Promo):** Setup for improving CTR by 0.3pp.
    *   **Mini-Case (Analysis):** Computing differences and checking guardrails.
    *   **Hands-on Tasks:** Computing sample size and running simulations.
    *   **Quiz:** Questions on Type I/II errors, p-values, and peeking.
    *   **Key Takeaways:** Randomization turns correlation into causation; power/MDE govern reliability.
    *   **References:** Recommended reading (Kohavi, Goodman, Deng).
    *   **Appendix:** Formulas and Python code for Two-Proportion Z-test and CUPED.

---

### **BA6: Regression for Business**
**Focus:** OLS, Regularization (Ridge/Lasso), and Interpretability (SHAP)

*   **Introduction & OLS**
    *   **Title/Objectives:** Translating business questions to regression, fitting OLS, applying regularization, and using SHAP.
    *   **Agenda:** Problem framing, OLS, Multicollinearity, Regularization, Importance, SHAP, and Case Study.
    *   **Running Case:** Telco monthly revenue prediction with levers like usage and pricing.
    *   **OLS Intuition:** Minimizing squared errors to find coefficients.
    *   **OLS Assumptions:** Linearity, Exogeneity, Homoscedasticity, and Normality.
    *   **Visual Example:** Scatter plot of Sales vs. Price with OLS line.
    *   **Estimation:** Closed-form solution $\beta = (X^TX)^{-1}X^Ty$.

*   **Diagnostics & Regularization**
    *   **Correlation Heatmap:** Visualizing predictor relationships.
    *   **Multicollinearity:** How correlated predictors inflate variance and destabilize coefficients.
    *   **Checks:** Using heatmaps and VIF; remedies like dropping features.
    *   **Regularization Overview:** The bias-variance trade-off.
    *   **Ridge (L2):** Squared magnitude penalty; shrinks coefficients smoothly; good for multicollinearity.
    *   **Lasso (L1):** Absolute magnitude penalty; induces sparsity (feature selection).
    *   **Elastic Net:** Combination of L1 and L2 penalties.
    *   **Choosing Lambda:** Using Cross-Validation to select regularization strength.
    *   **Model Evaluation:** Train/Test splits, RMSE/MAE, and Adjusted $R^2$.
    *   **Visuals:** Coefficient paths for Ridge and Lasso; MSE vs. Lambda plots.

*   **Interpretability & Feature Importance**
    *   **Prediction Intervals:** Communicating uncertainty around point forecasts (Visual).
    *   **Feature Importance Options:** Standardized coefficients, Permutation importance, and Partial Dependence.
    *   **SHAP Concepts:** Additivity, local accuracy, and attribution of prediction to features.
    *   **Case Questions:** Identifying revenue drivers and estimating lift from plan conversion.
    *   **Diagnostics Visuals:** Predicted vs. Actual, Residuals vs. Fitted, Residuals Histogram.
    *   **Transformations:** Log/Box-Cox for heavy tails; Binning for outliers.
    *   **Interactions:** capturing non-linear effects (e.g., Price $\times$ Ads).
    *   **Robust & Categorical:** Huber regression and dummy coding.
    *   **Diagnostics Checklist:** Specification, residuals, influence, and drift.

*   **Case Study & Wrap-up**
    *   **Feature Correlation:** Heatmap of Telco features.
    *   **Ridge Paths:** Visualizing coefficient shrinkage.
    *   **Permutation Importance:** Bar chart showing $\Delta MSE$ on test set.
    *   **Local Explanation:** Decomposing one prediction.
    *   **Local Contribution Plot:** Linear SHAP-like waterfall chart.
    *   **Code Snippets:** NumPy OLS, Ridge Path, and Permutation Importance.
    *   **Case Results:** RMSE interpretation and top drivers.
    *   **Hands-on:** Recreating OLS and tuning Ridge/Lasso.
    *   **Quiz:** Questions on assumptions, regularization choice, and scaling.
    *   **Takeaways:** Start simple, regularize for stability, explain for trust.
    *   **References:** ISLR, Hastie, Kuhn.
    *   **Appendix:** Matrix notes and Lasso geometry.

---

### **BA7: Classification Evaluation**
**Focus:** ROC/PR Curves, Thresholding, Calibration, and Cost-Sensitive Learning

*   **Fundamentals**
    *   **Title/Context:** Importance of evaluation beyond accuracy; asymmetric costs.
    *   **Agenda:** Confusion matrix, ROC/AUC, PR/AP, Thresholds, Cost-sensitive, Calibration.
    *   **Learning Objectives:** Choosing thresholds, incorporating costs, and assessing calibration.
    *   **Confusion Matrix:** Definitions of TP, FP, TN, FN.
    *   **Core Rates:** TPR (Recall), FPR, TNR, Precision (PPV).
    *   **F-measure:** $F1$ and $F_{\beta}$ scores for balancing precision and recall.
    *   **Accuracy Pitfalls:** Visual example of how accuracy fails with imbalance.
    *   **Advanced Metrics:** MCC (Matthews Correlation Coefficient) and Balanced Accuracy/G-Mean.

*   **Curves & AUC**
    *   **Score Distributions:** Visualizing overlapping histograms of positives/negatives.
    *   **ROC Curve:** Plotting TPR vs. FPR; AUC interpretation as ranking probability.
    *   **ROC Table:** Example calculation of rates at different thresholds.
    *   **Youden’s J:** Finding the optimal cut-off on the ROC curve.
    *   **Iso-Cost Lines:** Visualizing cost optimization on the ROC plot.
    *   **ROC Misleading:** Why ROC fails on severe imbalance; introduction to PR curves.
    *   **PR Curve:** Precision vs. Recall; baseline precision equals prevalence.
    *   **PR Table & AP:** Example calculations and Average Precision definition.
    *   **ROC vs. PR:** Guidelines on when to use which.

*   **Thresholding & Calibration**
    *   **Thresholding Strategies:** Maximizing F1, F-beta, or constraining Precision/Recall.
    *   **Business Framing:** Translating scores to decisions (Spam vs. Not Spam).
    *   **Calibration Importance:** Why probabilities must be accurate for ROI calculations (e.g., fraud risk).
    *   **Top-k/Quota:** Selection based on budget capacity rather than score threshold.
    *   **Cost-Sensitive Setup:** Defining the cost matrix (Cost of FP vs. Cost of FN).
    *   **Expected Cost:** Finding the threshold $\tau$ that minimizes total expected cost.
    *   **Cost Curves:** Visualizing cost across all thresholds.
    *   **Bayes Rule Threshold:** Formula $\tau = C_{FP} / (C_{FN} + C_{FP})$.
    *   **Risk & Regulations:** Setting constraints (max FPR).
    *   **Imbalanced Data:** Strategies like resampling and stratified CV.
    *   **Lift & Gain Charts:** Measuring model effectiveness against random selection.

*   **Diagnostics & Workflow**
    *   **Calibration Diagnostics:** Reliability diagrams, Brier score, Log loss, ECE.
    *   **When to Care:** Scenario table (Ranking vs. ROI calculation).
    *   **Reliability Diagram:** Visual example of perfect vs. actual calibration.
    *   **Calibration Impact:** Table showing metric improvements (Brier/ECE) post-calibration.
    *   **Cross-Validation:** Training calibrators on held-out data to avoid leakage.
    *   **Thresholding Effect:** Theoretical vs. Empirical threshold selection.
    *   **Workflow:** Train $\to$ Evaluate $\to$ Calibrate $\to$ Threshold $\to$ Back-test.
    *   **Pitfalls & Tips:** Avoiding leakage, monitoring drift, and communicating with stakeholders.
    *   **Checklist & Implementation:** Summary of steps and Python hints.

---

### **BA8: Segmentation & Dimensionality Reduction**
**Focus:** Clustering (K-Means, GMM), PCA, and Customer Segmentation (RFM)

*   **Introduction & Clustering**
    *   **Title/Agenda:** Segmentation, K-Means, GMM, PCA, RFM.
    *   **Objectives:** When to use clustering/PCA, implementing algorithms, and deriving segments.
    *   **Why Segmentation:** Personalization, resource prioritization, and structure discovery.
    *   **Data Prep:** Feature engineering, scaling (critical), and handling outliers.
    *   **Distance Metrics:** Euclidean (magnitude), Cosine (direction), Mahalanobis (covariance).
    *   **K-Means Intuition:** Minimizing Within-Cluster Sum of Squares (Inertia).
    *   **Algorithm Steps:** Initialize, Assign, Update, Iterate.
    *   **Initialization:** Random vs. K-Means++ to avoid local minima.
    *   **Scaling & Outliers:** Importance of StandardScaler and outlier trimming.
    *   **Distance Choices:** Using Cosine for sparse data; K-Prototypes for mixed data.
    *   **Failure Modes:** Non-convex shapes, varying densities, and unequal sizes.
    *   **Choosing k (Elbow):** Visualizing inertia to find diminishing returns.
    *   **Choosing k (Silhouette):** Measuring cohesion vs. separation.

*   **Advanced Clustering & GMM**
    *   **Mini-Batch K-Means:** Scaling to big data via random sampling.
    *   **Diagnostics:** Stability checks with random seeds and checking cluster sizes.
    *   **Case Study:** Customer segmentation using Usage, Tenure, and Spend.
    *   **Visuals:** 3D scatter of customer segments.
    *   **Naming & Action:** Creating personas (e.g., Power Users) and assigning KPIs/Actions.
    *   **GMM Intro:** Overcoming spherical assumptions of K-Means.
    *   **GMM Concept:** Modeling data as a mixture of Gaussian distributions (Mean, Covariance, Weight).
    *   **Covariance Types:** Spherical, Diagonal, Full, Tied.
    *   **EM Algorithm:** E-step (Soft assignment) and M-step (Update parameters).
    *   **Selecting Components:** Using AIC/BIC to penalize complexity.
    *   **Soft Assignments:** Probabilistic membership and strategic thresholds.
    *   **Comparison:** Table contrasting K-Means vs. GMM.
    *   **Visuals:** Comparison plots, density contours, and soft assignment sizing.

*   **Dimensionality Reduction (PCA)**
    *   **PCA Intro:** Principal Component Analysis.
    *   **Why Reduce:** Noise reduction, compression, visualization, and curse of dimensionality.
    *   **Variance & Covariance:** Orthogonal directions maximizing variance.
    *   **Steps:** Center data $\to$ Compute components (SVD) $\to$ Select $\to$ Transform.
    *   **Whitening:** Rescaling variance to unit sphere (optional).
    *   **Visuals:** 2D Projection and Biplots (Scores + Loadings).
    *   **Reconstruction Error:** Measuring information loss.
    *   **Interpreting Loadings:** Understanding feature influence via vectors and angles.
    *   **Caveats:** Linear limitations (Swiss Roll example), scale/outlier sensitivity.
    *   **Selection:** Scree Plot (Elbow) and Cumulative Explained Variance (90-95% threshold).
    *   **Beyond PCA:** Introduction to t-SNE and UMAP for non-linear manifolds.

*   **RFM & Implementation**
    *   **RFM Concept:** Recency, Frequency, Monetary value for behavioral segmentation.
    *   **Scoring:** Quantile binning (1-5 scale) and concatenation.
    *   **Taxonomy:** Standard segments (Champions, Loyalists, At Risk).
    *   **Dashboarding:** KPIs per segment, migration trends, and conversion tracking.
    *   **Distributions:** Handling skewness and the Pareto principle.
    *   **Visuals:** RFM Scatter with bubble size for Frequency.
    *   **Hybrid Pipeline:** Standardize $\to$ PCA $\to$ Clustering $\to$ Profiling.
    *   **Validation:** Time-based splits and monitoring drift.
    *   **Ethics:** Avoiding sensitive attributes and ensuring explainability.
    *   **Deployment:** Batch vs. Real-time scoring and feedback loops.
    *   **References & Summary:** Recommended reading and key takeaways.

---

### **BA9: Advanced Time Series Analysis & Forecasting**
**Focus:** Decomposition, Exponential Smoothing (ETS), ARIMA, and Validation

*   **Introduction & Fundamentals**
    *   **Title/Agenda:** Decomposition (STL), ETS, ARIMA, Holidays, and Cross-Validation.
    *   **Why Time Series:** Strategic planning vs. Operational efficiency vs. Risk management.
    *   **Statistical Shift:** Moving from independent random samples ($y_i \perp y_j$) to temporal dependence ($Cov(y_t, y_{t-1}) \neq 0$).
    *   **Data Cube Paradigm:** Univariate vs. Multivariate (Exogenous) vs. Hierarchical structures; frequency and granularity implications.

*   **Decomposition & Smoothing**
    *   **Components:** Trend-Cycle ($T_t$), Seasonality ($S_t$), and Remainder ($R_t$).
    *   **Decomposition Models:** Additive ($Y = T + S + R$) vs. Multiplicative ($Y = T \times S \times R$) assumptions.
    *   **STL Decomposition:** Using Loess for time-varying seasonality (unlike classical fixed seasonality).
    *   **Exponential Smoothing Philosophy:** Weighted averages where weights decay exponentially for older data ($W_j = \alpha(1-\alpha)^j$).
    *   **Simple Exponential Smoothing (SES):** For data with no trend/seasonality (ETS A,N,N); Level equation.
    *   **Holt’s Linear Trend:** Adding a trend equation ($b_t$); forecasts continue indefinitely.
    *   **Damped Trend:** introducing phi ($\phi$) to flatten long-term forecasts (ETS A,Ad,N) for realistic business planning.
    *   **Holt-Winters (Additive):** Recursive equations for Level, Trend, and Seasonal components.
    *   **Holt-Winters (Multiplicative):** Using division/multiplication for proportional seasonality (ETS M,A,M).
    *   **Model Selection:** The ETS taxonomy (Error, Trend, Seasonal combinations) and using AICc for selection.

*   **ARIMA & Stationarity**
    *   **ARIMA Intro:** Describing autocorrelations rather than structural components; Pillars: AR (p), I (d), MA (q).
    *   **Stationarity:** Constant mean, variance, and autocovariance over time; visual examples.
    *   **ADF Test:** Augmented Dickey-Fuller test for unit roots ($H_0$: Non-stationary).
    *   **Differencing (I):** First differencing ($d=1$) for trend removal; Seasonal differencing ($D=1$) for seasonality.
    *   **AR Models (p):** Regression on past values ("momentum").
    *   **MA Models (q):** Regression on past forecast errors ("shocks").
    *   **ACF & PACF:** Diagnostic plots to determine $p$ (PACF cutoff) and $q$ (ACF cutoff).
    *   **Box-Jenkins Procedure:** The 6-step workflow from plotting to diagnosing residuals.

*   **Advanced Techniques**
    *   **Moving Holidays:** Why standard SARIMA fails on Easter/Ramadan.
    *   **Dynamic Regression (ARIMAX):** Combining Linear Regression (for external factors) with ARIMA errors.
    *   **Dummy Variables:** Creating binary flags for holidays and window effects (lead-up/hangover).
    *   **Facebook Prophet:** Additive model using piecewise trends and Fourier terms; handles outliers and holidays easily.

*   **Validation & Best Practices**
    *   **Golden Rule:** Never shuffle time series data (peeking); use Walk-Forward Validation.
    *   **TSCV Strategies:** Expanding Window (uses all history) vs. Rolling Window (constant size, adapts to structural breaks).
    *   **MAPE Issues:** Undefined at zero, penalizes positive errors heavily.
    *   **MASE:** Mean Absolute Scaled Error; compares model error to a Seasonal Naive baseline (MASE < 1 is good).
    *   **Case Studies:**
        *   **Tourism:** Strong seasonality favors ETS.
        *   **Rossmann Sales:** Complex holidays and promos favor ARIMAX.
    *   **Summary Table:** Comparing ETS, ARIMA, and Prophet/Dynamic Regression.
    *   **Workflow:** 7-step process from Visualization to Selection.
    *   **Conclusion:** "All models are wrong, some are useful"; start simple.

---

### **BA10: Recommender Systems**
**Focus:** Collaborative Filtering, Matrix Factorization, and Ranking Metrics

*   **Business Context & Approaches**
    *   **Agenda:** Business Value, Content-Based, Collaborative Filtering, Evaluation.
    *   **Business Problem:** Information Overload and the Paradox of Choice causing churn.
    *   **The Solution:** Shifting from "Active Search" to "Passive Discovery" (Netflix/Amazon stats).
    *   **Long Tail Effect:** Helping users discover niche items to increase catalog utilization.
    *   **Feedback Data:** Explicit (Ratings) vs. Implicit (Clicks/Views); pros and cons of each.
    *   **Taxonomy:** Content-Based, Collaborative Filtering (User/Item), and Hybrid.

*   **Content-Based Filtering**
    *   **Case Study:** "MovieStream" data structure and objective.
    *   **Intuition:** Recommending items with similar features to what a user liked.
    *   **Item Profiles:** Converting structured data and text (TF-IDF) into vectors.
    *   **User Profiles:** Aggregating consumed item vectors.
    *   **Cosine Similarity:** Measuring the angle between User and Item vectors.
    *   **Workflow:** Fetch profile $\to$ Generate candidates $\to$ Score $\to$ Rank.
    *   **Pros/Cons:** Independence and transparency vs. Overspecialization (Filter Bubble) and cold start.

*   **Collaborative Filtering (CF)**
    *   **Wisdom of Crowds:** Relying on user behavior patterns rather than item metadata.
    *   **Memory-Based CF:** User-Based ("people like me") vs. Item-Based ("bought together"); Item-based stability.
    *   **Interaction Matrix:** The challenge of sparsity (>99% empty).
    *   **Implicit Feedback Challenges:** No negatives, noisy intensity; Solution: Confidence vs. Preference.
    *   **Confidence Model:** Mapping interaction counts ($r_{ui}$) to confidence weights ($c_{ui}$).

*   **Matrix Factorization (Model-Based)**
    *   **Concept:** Decomposing the sparse matrix $R$ into User ($U$) and Item ($V$) matrices ($R \approx U \times V^T$).
    *   **Latent Factors:** Hidden features (e.g., "Serious vs. Funny") learned by the machine.
    *   **Prediction:** Dot product of User and Item latent vectors.
    *   **ALS Training:** Alternating Least Squares for parallelizable training on implicit data.
    *   **Vector Example:** Visualizing User and Item vectors to show similarity.
    *   **Pros/Cons:** Serendipity vs. Cold Start and Popularity Bias.
    *   **Comparison Table:** Content-Based vs. Collaborative Filtering trade-offs.

*   **Evaluation**
    *   **Offline vs. Online:** Historical splitting vs. A/B testing.
    *   **Metric Categories:** Prediction Accuracy (RMSE) vs. Ranking Accuracy (Top-N).
    *   **RMSE:** Formula and limitation (users care about rank, not exact numbers).
    *   **Top-N Logic:** Hits and Misses in the top $K$ recommendations.
    *   **Precision & Recall @ K:** Usefulness vs. Coverage trade-off.
    *   **MAP:** Mean Average Precision to account for position.
    *   **NDCG:** Normalized Discounted Cumulative Gain; the gold standard for ranking.
    *   **Online/Business Metrics:** CTR, Conversion Rate, Diversity, and Novelty.
    *   **Advanced Topics:** Deep Learning (NCF), Sequence models (RNNs), and LLMs.
    *   **Python Ecosystem:** Libraries like Surprise, Implicit, and TensorFlow Recommenders.
    *   **Takeaways:** Shift to discovery, handling implicit data, and ranking metrics.

---

### **BA11: Storytelling with Data**
**Focus:** Narrative Structure, Frameworks (KPI Trees/Funnels/Cohorts), and Dashboard Design

*   **Context & Narrative**
    *   **Title/Agenda:** Executive's Dilemma, Storytelling, Frameworks, Dashboards, Metric Layer.
    *   **Executive's Dilemma:** "Data Rich but Insight Poor," Analysis Paralysis, Trust Issues.
    *   **Storytelling Triangle:** Intersecting Data, Visuals, and Narrative to drive Change.
    *   **Story Arc:** Context (Setup) $\to$ Conflict (Analysis) $\to$ Resolution (Action).
    *   **Signal vs. Noise:** Maximizing the Data-Ink Ratio (Tufte); removing clutter.
    *   **Audience Tailoring:** C-Level (Strategy/ROI) vs. Managers (Tactics) vs. Analysts (Root cause).
    *   **Insight Levels:** Moving from Descriptive ("What") to Prescriptive ("So What?"/Action).

*   **Analytical Frameworks**
    *   **Toolkit Overview:** KPI Trees, Funnels, Cohorts.
    *   **Metrics vs. KPIs:** "If everything is important, nothing is important".
    *   **KPI Driver Tree:** Decomposing goals mathematically (Additive/Multiplicative) for root cause analysis.
    *   **Tree Example (E-commerce):** GMV $\to$ Traffic $\times$ Conversion $\times$ Price.
    *   **Tree Storytelling:** Tracing a GMV drop to a specific conversion issue.
    *   **Funnel Analysis:** AIDA model; identifying bottlenecks and "Leaky Buckets".
    *   **Funnel Visualization:** Bar/Funnel charts showing absolute numbers and conversion rates.
    *   **Cohort Analysis:** The "Time Machine" for retention; avoiding aggregate masking.
    *   **Reading Heatmaps:** Diagonal (Current) vs. Vertical (Lifecycle) reading.
    *   **LTV Cohorts:** Accumulating revenue per user to compare against CAC.
    *   **Lagging vs. Leading:** Output (hard to change) vs. Input (actionable) indicators.
    *   **Dashboard Balance:** Recommended 60/40 ratio of Lagging to Leading metrics.
    *   **Summary:** Matching frameworks to problems (Diagnose, Optimize, Understand Behavior, Predict).

*   **Dashboard Design & Architecture**
    *   **Dashboard Types:** Strategic, Operational, Analytical.
    *   **5-Second Rule:** Scannability for immediate answers.
    *   **Layout Patterns:** F-Pattern and Z-Pattern; placing BANs (Big Area Numbers) top-left.
    *   **Color Semantics:** Using color for meaning (Good/Bad/Alert), not decoration.
    *   **Design Pitfalls:** Clutter, lack of context, wrong charts, scroll fatigue.
    *   **Metric Layer:** Solving "Metric Chaos" by defining metrics as code (e.g., dbt/Looker).
    *   **Modern Data Stack:** The role of the Metric Layer in governance and consistency.

*   **Application**
    *   **Case Study:** CSO Dashboard example (Revenue, Quota, Pipeline).
    *   **Visual Demo:** Highlighting context (vs. Last Year) and drill-downs.
    *   **Key Takeaways:** Story structure, Tool selection, Simplicity, and Architecture.
    *   **Homework:** KPI Driver Tree and Leading Indicator exercise.