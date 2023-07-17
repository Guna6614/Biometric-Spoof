function varargout = face_featurevector(varargin)
% FACE_FEATUREVECTOR MATLAB code for face_featurevector.fig
%      FACE_FEATUREVECTOR, by itself, creates a new FACE_FEATUREVECTOR or raises the existing
%      singleton*.
%
%      H = FACE_FEATUREVECTOR returns the handle to a new FACE_FEATUREVECTOR or the handle to
%      the existing singleton*.
%
%      FACE_FEATUREVECTOR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FACE_FEATUREVECTOR.M with the given input arguments.
%
%      FACE_FEATUREVECTOR('Property','Value',...) creates a new FACE_FEATUREVECTOR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before face_featurevector_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to face_featurevector_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help face_featurevector

% Last Modified by GUIDE v2.5 06-Oct-2014 18:30:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @face_featurevector_OpeningFcn, ...
                   'gui_OutputFcn',  @face_featurevector_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before face_featurevector is made visible.
function face_featurevector_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to face_featurevector (see VARARGIN)

% Choose default command line output for face_featurevector
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes face_featurevector wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = face_featurevector_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

load matlab2
global C

C = face_img;
C=imresize(C,[256 256]);
if size(C,3)==3,C=rgb2gray(C);end;
axes(handles.axes1)
imshow(C),title('FACE');
    
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global C filtered_3 

if ndims(C) == 3
    In3 = rgb2gray(C);
else
    In3 = C;
end

%% Gaussian Filtering

hsize = [3 3];
sigma = 0.5;
h = fspecial('gaussian', hsize, sigma);
filtered_3 = imfilter(In3,h);
axes(handles.axes1)
imshow(filtered_3),title('FILTERED FACE');

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

warning off all;
global C filtered_3 data

%% Read Original & Distorted Images

origImg = C;

distImg = filtered_3;

%% If the input image is rgb, convert it to gray image

noOfDim = ndims(origImg);
if(noOfDim == 3)
    origImg = rgb2gray(origImg);
end

noOfDim = ndims(distImg);
if(noOfDim == 3)
    distImg = rgb2gray(distImg);
end

%% Size Validation

origSiz = size(origImg);
distSiz = size(distImg);
sizErr = isequal(origSiz, distSiz);
if(sizErr == 0)
    disp('Error: Original Image & Distorted Image should be of same dimensions');
    return;
end

disp ('                                        ')
disp('Face Features');
disp ('                                        ')

%% Mean Square Error 

MSE = MeanSquareError(origImg, distImg);
disp('Mean Square Error = ');
MSE1 = abs(MSE);

disp(MSE);

%% Peak Signal to Noise Ratio 

PSNR = PeakSignaltoNoiseRatio(origImg, distImg);
disp('Peak Signal to Noise Ratio = ');
PSNR1 = abs(PSNR);
disp(PSNR1);

%% Normalized Cross-Correlation 

NK = NormalizedCrossCorrelation(origImg, distImg);
disp('MNormalized Cross-Correlation  = ');
NK1 = abs(NK);
disp(NK1);

%% Average Difference 

AD = AverageDifference(origImg, distImg);
disp('Average Difference  = ');
AD1 = abs(AD);
disp(AD1);


%% Structural Content 

SC = StructuralContent(origImg, distImg);
disp('Structural Content  = ');
SC1 = abs(SC);
disp(SC1);

%% Maximum Difference 

MD = MaximumDifference(origImg, distImg);
disp('Maximum Difference = ');
MD1 = abs(MD);
disp(MD1);

%% Normalized Absolute Error

NAE = NormalizedAbsoluteError(origImg, distImg);
disp('Normalized Absolute Error = ');
NAE1 = abs(NAE);
disp(NAE1);


%% Signal to signal noise ratio, SNR

noise = double(distImg) - double(origImg); % assume additive noise

%% check noise
noisyImageReconstructed = double(origImg) + noise;
residue = noisyImageReconstructed - double(distImg);

if (sum(residue(:) ~= 0))
    disp('The noise is NOT relevant.');
end

snr_power = SNR(origImg, noise);
    
    disp('Signal to noise ratio = ');
disp(snr_power);
snr1 = abs(snr_power);
    
%% Laplacian MSE

A = double(origImg);
B = double(distImg);
OP=4*del2(A);
LMSE=sum(sum((OP-4*del2(B)).^2))/sum(sum(OP.^2));
% fprintf('LMSE (Laplacian Mean Squared Error) = %f\n',LMSE);

LMSE1 = abs(LMSE);
disp('LMSE (Laplacian Mean Squared Error) = ');
disp(LMSE1);

%% R-Averaged MD

R = 10;
RAMD = MD/R;
RAMD1 = abs(RAMD);
disp('R-Averaged MD = ');
disp(RAMD1);

%% Mean Angle Similarity 

 CosTheta = dot(A,B)/(norm(A)*norm(B));

ThetaInDegrees = acos(CosTheta)*180/pi;

theta = 1 - (mean(mean(ThetaInDegrees)));
MAS1 = abs(theta);

%% Mean Angle Magnitude Similarity

mm = 1 - theta;
m5 = 1 - CosTheta/255;
MAMS1 = 1 - (mm .* m5);
MAMS = mean(mean(MAMS1));
MAMS1 = abs(MAMS);

%% Total Edge Difference

[M,N] = size(A);
e1 = edge(A,'sobel');
e2 = edge(B,'sobel');

edge_diff = e1 - e2;
TED = sum(sum(edge_diff)) / (M .* N);
disp('Total Edge Difference = ');
TED1 = abs(TED);
disp(TED1);

%% Total corner difference

corners = detectHarrisFeatures(A);
corners1 = detectHarrisFeatures(B);

im = A;
im1 = B;
sigma = 2;
radius = 3;
thresh = 100;
disp1 = 1;
[cim, r, c] = harris(im, sigma, thresh, radius, disp1);
axes(handles.axes1); imagesc(im), axis image, colormap(gray), hold on
	    plot(c,r,'ys'), title('corners detected on Input image');pause(1);
[cim1, r1, c1] = harris(im1, sigma, thresh, radius, disp1);
axes(handles.axes1); imagesc(im), axis image, colormap(gray), hold on
	    plot(c,r,'ys'), title('corners detected on filtered image');pause(1);
no_cor_I = mean(mean(r));
no_cor_I1 = mean(mean(r1));
max_I = max(r);
max_I1 = max(r1);

nn = no_cor_I - no_cor_I1;
TCD = nn ./ (max_I .* max_I1);

TCD1 = abs(TCD);
disp('Total Corner Difference = ');
disp(TCD1);

axes(handles.axes1);imshow(A,[]); hold on;plot(corners.selectStrongest(50));title('Original Image');pause(1);
axes(handles.axes1);imshow(B,[]); hold on;plot(corners1.selectStrongest(10));title('Distorted Image');pause(1);

%% Spectral magnitude error

F1 = fft2(A);
f2 = ifft2(F1);
F1 = fft2(B);
f3 = ifft2(F1);

axes(handles.axes1);imshow(F1);title('DFT on original image');pause(1);
axes(handles.axes1);imshow(f2,[]);title('IDFT on original image');pause(1);
axes(handles.axes1);imshow(F1);title('DFT on distorted image');pause(1);
axes(handles.axes1);imshow(f3,[]);title('IDFT on original image');pause(1);
sp = sum(sum(abs((F1 - F1)).^2));
sp1 = mean(mean(sp));
SME = sp1 ./ (M .* N);
SME1 = abs(SME);
disp('Spectral Magnitude Error = ');
disp(SME1);

%% Spectral Phase Error

fftA = fft2(double(A));
fftB = fft2(double(B));

axes(handles.axes1);imshow(abs(fftshift(fftA)),[24 100000]), colormap gray
title('Image A FFT2 Magnitude');pause(1);
axes(handles.axes1), imshow(angle(fftshift(fftA)),[-pi pi]), colormap gray
title('Image A FFT2 Phase');pause(1);
axes(handles.axes1), imshow(abs(fftshift(fftB)),[24 100000]), colormap gray
title('Image B FFT2 Magnitude');pause(1);
axes(handles.axes1), imshow(angle(fftshift(fftB)),[-pi pi]), colormap gray
title('Image B FFT2 Phase');pause(1);

fftC = abs(fftA).*exp(i*angle(fftB));
fftD = abs(fftB).*exp(i*angle(fftA));

s1 = angle(F1);
s2 = angle(F1);
s1 = abs(s1);
s2 = abs(s2);
s11 = max(max(s1));
s12 = max(max(s2));

sp = (s11 - s12).^2;
SPE = sp ./ (M .* N);
SPE1 = abs(SPE);
disp('Spectral Phase Error = ');
disp(SPE1);



%% Gradient Magnitude Error & Gradient Phase Error

[Gx, Gy] = imgradientxy(A);
[Gmag, Gdir] = imgradient(Gx, Gy);

GME5 = mean(mean(Gmag));
GME1 = abs(GME5);
GPE5 = mean(mean(Gdir));
GPE1 = abs(GPE5);

[Gx1, Gy1] = imgradientxy(B);
[Gmag1, Gdir1] = imgradient(Gx1, Gy1);
GME2 = Gmag1(100);
GME11 = abs(GME2);
GPE2 = Gdir1(150);
GPE11 = abs(GPE2);

m5 = GME1;
m6 = GPE1;

disp('Gradient Magnitude Error = ');
disp(m5);

disp('Gradient Phase Error = ');
disp(m6);


%% SSIM

img1 = A;img2 = B;
K = [0.05 0.05];
window = ones(8);
L = 100;

[mssim, ssim_map] = ssim(img1, img2, K, window, L);
axes(handles.axes1);imshow(max(0, ssim_map).^4);title('SSIM index map');pause(1);

mssim1 = abs(mssim);
disp('Structural Similarity Index Measure (SSIM) = ');
disp(mssim1);


%% FR-IQMs: Information Theoretic Measures:

%% Visual Information Fidelity (VIF)

img1 = A;
hsize = [3 3];
sigma = 0.5;
h = fspecial('gaussian', hsize, sigma);
img2 = B;

vif = vifp_mscale(img1, img2);
disp('Visual Information Fidelity (VIF) = ');
vif1 = abs(vif);
disp(vif1);

%% Reduced Reference Entropic Difference index (RRED)

rred1 = compute_rred(A,B);
disp('Reduced Reference Entropic Difference index (RRED)  = ');
disp(rred1);

%% No-Reference IQ Measures

%% JPEG Quality Index (JQI)

Quality_Score = jpeg_quality_score(A);

disp('JPEG Quality Index (JQI)  = ');
disp(Quality_Score);

%% High-Low Frequency Index (HLFI)

%% Lower frequency

img   = fftshift(A);
F     = fft2(img);

%% Power Spectrum computation 

im_Pfft = abs(F.^2);

%% log of power, avoid log of zeros

im_logPfft = log(im_Pfft+eps);

%% Upper frequency

img1   = fftshift(B);
F1     = fft2(img1);

%% Power Spectrum computation 

im_Pfft1 = abs(F1.^2);

%% log of power, avoid log of zeros

im_logPfft1 = log(im_Pfft1+eps);

hl = im_logPfft - im_logPfft1;

HLFI = mean(mean(hl));

HLFI1 = abs(HLFI);
disp('High-Low Frequency Index (HLFI)  = ');
disp(HLFI1);

%% Blind Image Quality Index (BIQI)

X = A;
N = 2;
theta = 10;
units = 'radian';
window_shape = 'square';
nod = 2;
firstangle = 20;
angleunits = 'radian';

[Qg,Qch]=blindimagequality(X,N,nod,firstangle,angleunits);

BIQI = Qg+Qch;
BIQI1 = abs(BIQI);
disp('Blind Image Quality Index (BIQI)  = ');
disp(BIQI1);

%% Natural Image Quality Evaluator (NIQE)

if ndims(A) == 3
    image = rgb2gray(A); 
else
    image = A;
end

imref = image;
load modelparameters.mat

blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;
NIQE = computequality(imref,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
    mu_prisparam,cov_prisparam)
NIQE1 = abs(NIQE);
disp('Natural Image Quality Evaluator (NIQE)  = ');
disp(NIQE1);

data = [MSE1;PSNR1;NK1;AD1;SC1;MD1;NAE1;snr1;LMSE1;RAMD1;MAS1;MAMS1;TED1;TCD1;SME1;SPE1;mssim1;vif1;rred1;HLFI1;BIQI1;NIQE1;m5;m6]/100;
data = sum(data)
% save data6 data6

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global data

load net1
y = round(sim(net1,data))

if y == 0
    msgbox('Fake');
elseif y == 1
    msgbox('Real');
else
    msgbox('Not Valid Input');
end
