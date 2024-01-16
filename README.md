# Uncovering Patterns and Predicting Chronic Disease Risks in the U.S​
​

## Here is the link to the Tableau interactive dashboard [Tableau - AirBbB in NYC Analytics](https://public.tableau.com/app/profile/nguyen.ho2733/viz/Airbnb_16840158070070/Dashboard3)

## 1. Problem Statement
I aim to use data mining algorithms to tackle a challenge in U.S. public health through classification of risk stratification and predictive models of chronic disease trends across different geographies from 2001 to 2021. The dataset `U.S. Chronic Disease Indicators (CDI)` was obtained to discover patterns and potential outbreak risks of chronic diseases at the state level. Through this work, I deliver the insight to improve the public health outcomes and preventive measures. 
The goals are to develop a predictive model capable of assessing the mortality associated with the disease and durations. I am also applying tree-based ensemble algorithms to classify the risk level of chronic disease, so it could reveal the most current potential outbreaks in the health landscape. These approaches serve to enhance strategic decision-making in public health, facilitating targeted interventions and resource allocation for disease prevention and management. 


## 2. Introduction to the dataset

For the project, I use the [U.S. Chronic Disease Indicators (CDI) Data](https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi) which offers a set of surveillance indicators developed by consensus among CDC, the Council of State and Territorial Epidemiologists (CSTE), and the National Association of Chronic Disease Directors (NACDD). These indicators are a wide range of indicators for the surveillance of chronic diseases, conditions, and risk factors at the state level that are essential for the evaluation of public health interventions. 
There are more than 900K data rows in this dataset, which more than 24 indicators over 50 states that includes 17 topic groups: alcohol; arthritis; asthma; cancer; cardiovascular disease; chronic kidney disease; chronic obstructive pulmonary disease; diabetes; immunization; nutrition, physical activity, and weight status; oral health; tobacco; overarching conditions; and new topic areas that include disability, mental health, older adults, reproductive health.
This dataset features indicators over 20 years of public health in the U.S., with 15 variables:

- `YearStart`:
- `YearEnd`: 
- `LocationAbbr`: 
- `DataSource`:
- `Topic`: 
- `Question`: 
- `DataValueUnit`:
- `DataValueType`: 
- `DataValue`
- `DataValueAlt`
- `StratificationCategory1`
- `Stratification1`
- `LocationID`
- `TopicID`
- `QuestionID`
- `DataValueTypeID`
- `StratificationCategoryID1`: the price for the listing per night
- `StratificationID1`: the minimum number of nights required for a booking 
- `Longitude`: the longitude coordinate of patient location
- `Latitude`: the latitude coordinate of the patient location

## 3. Data Exploration
<img width="1141" alt="Screen Shot 2023-05-14 at 10 19 46 PM" src="https://github.com/ChloeHo12/New-York-City-Airbnb-Price-Prediction/assets/98048503/4f2a70cc-ce0d-4338-b941-1852f40e9bd5">


## 4. Modeling
Followed by Data Processing, I employed random forest to classify risk level of chronic diseases and use it to predict the future patient status.

## 4. Conclusion
<img width="1141" alt="Screen Shot 2023-05-14 at 10 19 46 PM" src="https://github.com/ChloeHo12/New-York-City-Airbnb-Price-Prediction/assets/98048503/4f2a70cc-ce0d-4338-b941-1852f40e9bd5">

- `Low` and `Moderate` risk level, my model achieves perfect precision and recall on `Low` and nearly perfect on `Medium` which means the model can predict these classes without errors. The `High` risk level has slightly lower precision and recall but still performs fine. The `Very High` risk one has perfect precision, however, there are significant low for recall (0.33) which my model can correctly predict this class at 33% for true label. Its F1 score is also really low for this class compared to other classes, so it’s not-balance between precision and recall.  
- The accuracy for random forest classifier 0.9999790308916904 (nearly perfect) is impressive. However, I think it might not be an ideal model since we have an unbalanced dataset, so this metric can be misleading. The `Low` class has a majority that outnumbers the other classes. The `Very High` risk level has only 3 instances, it is underrepresented in our data. I think this is why it has low recall.
 

It's important to consider the strengths and limitations of each model, as well as the specific needs and goals of the analysis. For example, linear regression with best subset variable selection is a simpler and more interpretable model, while ensemble models like random forests or boosting can capture complex interactions and nonlinear relationships between predictors. The choice of the best model ultimately depends on various factors, highlighting the need to carefully consider model complexity and interpretability when selecting a model for a specific analysis.
