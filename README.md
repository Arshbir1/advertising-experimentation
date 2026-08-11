# Advertising Campaign Experimentation

An experimental framework for evaluating the effectiveness of an online advertising campaign using statistical hypothesis testing, sequential analysis, and machine-learning-based evaluation.

## Overview

The goal of this project is to analyze whether exposure to an advertising campaign produces a statistically significant change in user responses.

The project explores multiple approaches to experimental analysis:

- Exploratory data analysis
- Classical A/B hypothesis testing
- Sequential experimentation
- Machine-learning-based analysis

The emphasis is on understanding statistical significance, treatment effects, and the limitations of different experimental approaches.

## Dataset

The analysis uses an advertising campaign dataset containing information about users assigned to control and exposed groups.

The primary outcome represents whether a user correctly recalled the advertised brand.

The dataset is not included in this repository.

Place the dataset at:

    data/ABAdRecall.csv

before running the notebooks.

## Methodology

### 1. Exploratory Data Analysis

The first stage investigates the experimental population and identifies patterns in the control and exposed groups.

The analysis includes:

- Group distributions
- Response distributions
- User-level characteristics
- Campaign exposure patterns
- Potential differences between experimental groups

### 2. Classical A/B Testing

A classical hypothesis-testing framework is used to compare the control and treatment groups.

The analysis evaluates:

- Null and alternative hypotheses
- Observed differences between groups
- Statistical significance
- P-values
- Confidence intervals
- Campaign lift

The objective is to determine whether the observed difference is unlikely to have occurred by chance.

### 3. Sequential Testing

Sequential analysis explores how experimental conclusions can change as additional observations become available.

Instead of evaluating the experiment only after collecting a fixed sample, the analysis examines statistical evidence at different stages of the experiment.

This provides insight into:

- Statistical evidence over time
- Stability of experimental conclusions
- Potential early stopping decisions
- The risks associated with repeatedly checking experiment results

### 4. Machine Learning Analysis

Machine-learning models are used as a complementary analysis to investigate relationships between user characteristics, campaign exposure, and the observed response.

The ML analysis focuses on:

- Feature relationships
- Predictive performance
- Feature importance
- Comparison between experimental groups

Machine learning is treated as a predictive analysis rather than a replacement for causal experimental design.

## Experimental Pipeline

    Advertising Dataset
            |
            v
    Exploratory Data Analysis
            |
        +---+---+---+
        |       |   |
        v       v   v
    Classical Sequential ML
      A/B Test   Analysis Analysis
        |       |   |
        +---+---+---+
            |
            v
    Experimental Insights

## Repository Structure

    advertising-experimentation/
    |
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_classical_ab_test.ipynb
    │   ├── 03_sequential_testing_exploratory.ipynb
    │   └── 04_ml_analysis.ipynb
    |
    ├── scripts/
    │   ├── eda.py
    │   ├── classical_ab_test.py
    │   └── ml_analysis.py
    |
    ├── data/
    │   └── README.md
    |
    ├── .gitignore
    ├── requirements.txt
    └── README.md

## Key Concepts

- A/B Testing
- Statistical Hypothesis Testing
- Experimental Design
- Sequential Testing
- Confidence Intervals
- Statistical Significance
- Treatment and Control Groups
- Campaign Lift
- Feature Analysis
- Machine Learning

## Technologies

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Jupyter Notebook
- Matplotlib

## Running the Project

### 1. Clone the Repository

    git clone https://github.com/Arshbir1/advertising-experimentation.git
    cd advertising-experimentation

### 2. Install Dependencies

    pip install -r requirements.txt

### 3. Add the Dataset

Place the dataset at:

    data/ABAdRecall.csv

### 4. Run the Notebooks

Start Jupyter Notebook:

    jupyter notebook

Execute the notebooks in the following order:

    01_eda.ipynb
    02_classical_ab_test.ipynb
    03_sequential_testing_exploratory.ipynb
    04_ml_analysis.ipynb

## Results

The project evaluates the effectiveness of advertising exposure using multiple statistical and predictive approaches.

The analysis focuses on:

- Measuring differences between treatment and control groups
- Quantifying campaign lift
- Assessing statistical significance
- Investigating experimental results under sequential evaluation
- Understanding predictive relationships between campaign exposure and user responses

## Limitations

The results of observational or predictive analyses should not automatically be interpreted as causal effects.

In particular:

- Statistical significance does not necessarily imply practical significance.
- Repeatedly checking an experiment can affect false-positive rates.
- Machine-learning models identify predictive relationships rather than establishing causality.
- Experimental conclusions depend on the quality and design of the underlying treatment and control groups.

## License

This project is intended for educational and experimental purposes.
