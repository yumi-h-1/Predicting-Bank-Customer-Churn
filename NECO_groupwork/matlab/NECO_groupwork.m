%% Training MLP Model %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear command, workspace, and figures
clc;
clear;
close all;

%% Load data
X_test = readtable(['X_test.csv']);
Y_test = readtable(['Y_test.csv']);
X_train = readtable(['X_train.csv']);
Y_train = readtable(['Y_train.csv']);
X = readtable(["X.csv"])
Y = readtable(["Y.csv"])

%% Table to double
X_test = table2array(X_test);
Y_test = table2array(Y_test);
X_train = table2array(X_train);
Y_train = table2array(Y_train);
X = table2array(X);
Y = table2array(Y);

%%
