# Health Care Analytics

## **Predicting length of stay for a patient in a hospital**

### **Problem Statement**

This project was done for Janatahack, a 7 day data science competition organised by Analytics Vidhya. The task is a multiclass prediction problem to accurately predict the Length of Stay for each patient on case by case basis so that the Hospitals can use this information for optimal resource allocation and better functioning. The length of stay is divided into 11 different classes ranging from 0-10 days to more than 100 days.

The train dataset contains features related to patient, hospital and Length of stay on case basis. The scoring method used for this competition was 100 * Accuracy.

### **Solution Method**

After Exploratory Data Analysis, some features were added. After trying out different models like Xgboost, Bagging Classifier, Random Forest Classifier etc. LGBM Classifier was used for prediction as the best score was found from this method. 

### **Result**

The model produced a score of 43.672 on our test dataset and the score on the final benchmark dataset was 42.889.

***Note:*** Some more feature engineering can be done to further improve the score.
