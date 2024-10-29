-------------------------------------------------------------------------
INM427 Neural Computing | Yumi Heo | Student ID: 230003122 
-------------------------------------------------------------------------

=== Environment & Package Spec ===

*Jupyternotebook: 6.4.8
*Python version==3.9.12
*joblib==1.2.0
*pandas==1.4.2
*numpy==1.24.3
*matplotlib==3.7.1
*torch==2.0.1
*sklearn==1.3.0

=== Description ===

[Setup instructions for test]

1. Extract files from the 'Code & Test Set' zip file.
2. Please make sure the extracted files are in the same folder.
3. Using Anaconda Prompt, install each version of packages above.
4. Open Jupyter Notebook from Anaconda Prompt.
5. Select the folder where all the extracted files are stored.
6. For the test, open the code file named 'INM427 Neural Computing_Individual Project(Model
Test)_230003122_Yumi Heo.ipynb'.
7. Run all the code blocks to test the MLP and SVM models.

[File details]

-- Code
*INM427 Neural Computing_Individual Project(Model Training)_230003122_Yumi Heo.ipynb
-> This is the code file for initial data analysis, raining, tuning and selection of models.
*INM427 Neural Computing_Individual Project(Model Test)_230003122_Yumi Heo.ipynb
-> This is the code file for testing models.

-- Dataset
*X_test_for_MLP.pth
-> This is the features of the test set for the final MLP model.
*y_test_for_MLP.pth
-> This is the target of the test set for the final MLP model.
*X_test_for_SVM.csv
-> This is the features of the test set for the final SVM model.
*y_test_for_SVM.csv
-> This is the target of the test set for the final SVM model.

-- Model
*best_mlp_model.pth
-> This is the final MLP model.
*best_svm_model.joblib
->This is the final SVM model.
