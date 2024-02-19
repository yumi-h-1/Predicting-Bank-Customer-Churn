%% Training MLP Model %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear command, workspace, and figures
clc;
clear;
close all;

%% Load data
X = readtable(["X_array.csv"]);
Y = readtable(["y_array.csv"]);
X = table2array(X);
Y = table2array(Y);

%% Split the data into training and test sets ** 확인필요
rng("default") % For reproducibility
c = cvpartition(X,"Holdout",0.20);
trainingIndices = training(c); % Indices for the training set
testIndices = test(c); % Indices for the test set

%% 확인필요
Mdl = 
  ClassificationNeuralNetwork
             ResponseName: 'Y'
    CategoricalPredictors: []
               ClassNames: {'b'  'g'}
           ScoreTransform: 'none'
          NumObservations: 246
               LayerSizes: [35 20]
              Activations: 'sigmoid'
    OutputLayerActivation: 'sigmoid'
                   Solver: 'Adam'