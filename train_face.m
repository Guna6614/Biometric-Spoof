clc
clear all
close all
warning off all

%% load the data

load data1
load data2
load data3
load data4
load data5
load data6

T = [data1 data2 data3 data4 data5 data6];
x = [0 0 0 1 1 1];

%% create a feed forward neural network

net1 = newff(minmax(T),[30 20 1],{'logsig','logsig','purelin'},'trainrp');
net1.trainParam.show = 1000;
net1.trainParam.lr = 0.04;
net1.trainParam.epochs = 7000;
net1.trainParam.goal = 1e-5;

%% Train the neural network using the input,target and the created network

[net1] = train(net1,T,x);

%% save the network

save net1 net1

%% simulate the network for a particular input

y = round(sim(net1,T))

