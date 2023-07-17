
clc;clear all;close all;warning off all;

%% finger print

[f,p] = uigetfile('./fingerprint/*.jpg;*.bmp','Select Fingerprint Image'); 
finger_img = imread([p f]);
finger_img = imresize(finger_img,[256 256]);

%% iris 

[f,p] = uigetfile('./iris images/*.jpg;*.bmp','Select Iris Image'); 
iris_img = imread([p f]);
iris_img=imresize(iris_img,[256 256]);

%% face

[f,p] = uigetfile('./face/*.jpg;*.bmp','Select Face Image'); 
face_img = imread([p f]);
face_img=imresize(face_img,[200 200]);


if ndims(finger_img) == 3
    msgbox('Not a valid Input');
elseif ndims(finger_img) < 3
  fingerprint_featurevector
elseif ndims(iris_img) == 3
    msgbox('Not a valid Input');
elseif ndims(iris_img) < 3
    iris_featurevector
elseif ndims(face_img) == 3
    face_featurevector    
elseif ndims(face_img) < 3    
    msgbox('Not a valid Input');
end





