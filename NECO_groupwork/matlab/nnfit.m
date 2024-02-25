%% Training MLP Model %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear command, workspace, and figures
clc;
clear;
close all;

%% For reproducibility
rng("default")

%% Load data
X_train = readtable(['X_train.csv']);
y_train = readtable(['y_train.csv']);
X_val = readtable(['X_val.csv']);
y_val = readtable(['y_val.csv']);
X_test = readtable(['X_test.csv']);
y_test = readtable(['y_test.csv']);

%% Table to double
X_train = table2array(X_train);
y_train = table2array(y_train);
X_val = table2array(X_val);
y_val = table2array(y_val);
X_test = table2array(X_test);
y_test = table2array(y_test);

%% Build a model
Mdl = fitcnet(X_train,y_train)

%%
testAccuracy = 1 - loss(Mdl,X_test,y_test,"LossFun","classiferror")

%%
confusionchart(y_test,predict(Mdl,X_test))