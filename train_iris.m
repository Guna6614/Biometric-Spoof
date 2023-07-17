clc
clear all
close all
warning off all

%% load the data

load data_ir1
load data_ir2
load data_ir3
load data_ir4
load data_ir5
load data_ir6

T = [data_ir1 data_ir2 data_ir3 data_ir4 data_ir5 data_ir6];
x = [5 5 5 6 6 6];

%% create a feed forward neural network

net3 = newff(minmax(T),[30 20 1],{'logsig','logsig','purelin'},'trainrp');
net3.trainParam.show = 1000;
net3.trainParam.lr = 0.04;
net3.trainParam.epochs = 7000;
net3.trainParam.goal = 1e-5;

%% Train the neural network using the input,target and the created network

[net3] = train(net3,T,x);

%% save the network

save net3 net3

%% simulate the network for a particular input

y2 = round(sim(net3,T))

