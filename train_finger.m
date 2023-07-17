clc
clear all
close all
warning off all

%% load the data

load data_fin1
load data_fin2
load data_fin3
load data_fin4
load data_fin5
load data_fin6

T = [data_fin1 data_fin2 data_fin3 data_fin4 data_fin5 data_fin6];
x = [3 3 3 4 4 4];

%% create a feed forward neural network

net2 = newff(minmax(T),[30 20 1],{'logsig','logsig','purelin'},'trainrp');
net2.trainParam.show = 1000;
net2.trainParam.lr = 0.04;
net2.trainParam.epochs = 7000;
net2.trainParam.goal = 1e-5;

%% Train the neural network using the input,target and the created network

[net2] = train(net2,T,x);

%% save the network

save net2 net2

%% simulate the network for a particular input

y1 = round(sim(net2,T))

