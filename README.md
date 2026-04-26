# Machine-Learning-Based-Prediction-of-Groundwater-Transmissivity-Eastern-Region-of-Thailand-Area

  🚀 HydroML: Groundwater Transmissivity PredictionThis project implements a machine learning pipeline to estimate Transmissivity (T) from pumping test data. It utilizes the XGBoost algorithm to handle complex geological patterns and provides a set of tools for model evaluation and data reconstruction.
  
  📋 Project OverviewThe core objective is to predict hydraulic parameters in Eastern Thailand (Chonburi, Rayong, etc.) by training on historical pumping test records. The pipeline includes automated data augmentation (100x) and log transformation to handle the wide range of T values.
  
  📂 Repository Structure
  01_Fullpipline.py: The main machine learning engine.
    Performs data cleaning and log-transformation (log_T).
    Augments data with 100x copies and random jitter to improve robustness.
    Automates hyperparameter tuning across different tree depths.
  02_plt.py: Evaluation and visualization script.
    Generates Predicted vs. Actual plots.Creates Residual Plots and Error Distribution histograms.
    Visualizes Feature Importance.
  03_Reconstruct.py: Data reconstruction tool.Merges predicted results back with original metadata (e.g., UTM coordinates and FID).
  
  📊 Dataset: Column RequirementsTo run the pipeline, your input text file  should contain the following columns:
  Input Features
    Pumpingrat: The pumping rate (Q) measured during the test.
    Drawdown_m: The measured drop in water level (s).
    UNIT: The geological rock unit (e.g., Permian Carbonate).Depthofpip: The depth of the well pipe or screen.
    Specific_Capacity: An engineered feature (Q/s) calculated by the script.Target 
    VariableAverage_T_: The field-measured Transmissivity (T) used for training.
    Spatial Metadata (For Reconstruction) UTMEasting / UTMNorthin: Used by the reconstruction script for mapping.
    FID: Unique identifier for each well.
    
  🛠️ How to Use
    1.Preparation: Ensure your data is in a .txt or .csv format and update the paths in the CONFIG section of each script.
    2.Training: Run python 01_Fullpipline.py to train the model and save the best weights.
    3.Visualization: Run python 02_plt.py to analyze model performance and feature importance.
    4.Reconstruction: Run python 03_Reconstruct.py to generate a full report with all original columns and predictions.
    
  🔮 Future Development
    Data Expansion: Increasing real-world samples to reduce reliance on data augmentation.
    Temporal Study: Extending the data range to 20 or 30 years to better understand long-term groundwater movement.
    Scalability: This prototype can be retrained for other study areas provided there is enough field data.
    Developed by: Panpruek Pruekthikanee, Praeploy Premprasert,Warittha Srinuanon Department of Geology, Chulalongkorn University
