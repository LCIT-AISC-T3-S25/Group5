# Prompts

| Name      | Prompt Description                                                                                                                                                          |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Indraja   | - I have a deep learning model trained on grayscale images. How can I implement LIME?                                                                                       |
|           | - Code to print top contributing regions for all classes.                                                                                                                   |
|           | - Code to implement LIME on an SVM model trained on images.                                                                                                                 |
|           | - The model is trained on flattened RGB images (224×224×3). How to implement LIME to highlight important regions and display both the original and LIME result side by side.|
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Mukul Garg|-extract tar file jupyter                                                                                                                                                    |   
|           |-count files in a folder                                                                                                                                                     |
|           |-import tarfile                                                                                                                                                              |
|           | tar_path = './Yelp_Photos/yelp_photos.tar'                                                                                                                                  |
|           | extract_path = './extracted_files/'  # Replace or leave as-is for current folder                                                                                            |
|           | # Open and extract                                                                                                                                                          |      
|           | with tarfile.open(tar_path) as tar:                                                                                                                                         |
|           |    tar.extractall(path=extract_path)                                                                                                                                        |
|           |    print(f"Extracted all files to '{extract_path}'")
|           | extract only 1000 file
|Sri Datta N|-I’m using a KNN model on PCA-reduced features for classification. I already have X_train, X_test, y_test, and a best_knn model selected using GridSearchCV.Explain a single  |           |
|           | prediction using 10 most important PCA components.
|           |-Visualize the feature importances for the prediction
|           |-I’ve performed three rounds of model tuning: KNN with StandardScaler and hyperparameter tuningKNN with PCA (50–200 components)KNN with different scalers (Standard, MinMax, 
|           | Robust) For each round, show me:AccuracyAUC per classConfusion MatrixClassification Report Save all results to a dictionary (results) and report_dicts.
|           |-Generate training vs validation loss and accuracy curves for each tuning iteration of my CNN model
|           |-After each tuning iteration of my CNN, show a confusion matrix on the test set. Analyze improvements in class separability, especially for frequently confused classes
------------------------------------------|
| Resti Angelo Cenia|Having a DNN Model what is the model tuning plan                                                                                                                                                    |   
|           |based on this model tuning criteria class separability, lime used, analysis model performance, 3 rounds of empirical tuning                                                                                                                                                     |
|           |AUC calculation error fix                                                                                                          
|thejaswee  |- I’ve trained a KNN image classifier using Keras and I want to interpret one of its predictions using LIME.
|           |- For each tuning round of my CNN, use LIME to explain predictions on the same test image. Compare the explanations visually to show how model interpretability improved
|Gayathri  |- How to reduce datasize for SVM model training
|          | - how to convert X_train and Y_train to same datatype?
|          | -what is prob function and decicion function  and when do we use them?
