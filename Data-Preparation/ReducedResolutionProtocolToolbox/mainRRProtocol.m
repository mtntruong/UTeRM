%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 REDUCED RESOLUTION (RR) PROTOCOL TOOLBOX 
% 
% Please, refer to the following paper:
% G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", 
% IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2021.
% 
% % % % % % % % % % % % % 
% 
% Version: 1
% 
% % % % % % % % % % % % % 
% 
% Copyright (C) 2021
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;
clc;

%% Load full data
name = 'Test';
load(sprintf('%s_FR.mat',name));

%% Load parameters
%%% Interpolator flag (if equal to 1 bicubic interpolator; otherwise polynomial interpolator) 
bicubic = 0;

% Name of the acquisition sensor
sensor = 'WV3';

%%% Cut final image
flag_cut_bounds = 1;
dim_cut = 21;

%%% Threshold values out of dynamic range
thvalues = 0;

%%% Print Eps
printEPS = 0;

%%% Radiometric Resolution
L = 11;

%% GT
I_GT = I_MS_LR;

%% %%%%%%%%%%%%%    Preparation of image to fuse            %%%%%%%%%%%%%%
[I_MS_LR, I_PAN] = resize_images(I_MS_LR,I_PAN,ratio,sensor);

%% Upsampling
if bicubic == 1
    H = zeros(size(I_PAN,1),size(I_PAN,2),size(I_MS_LR,3));    
    for idim = 1 : size(I_MS_LR,3)
        H(:,:,idim) = imresize(I_MS_LR(:,:,idim),ratio);
    end
    I_MS = H;
else
    I_MS = interp23tap(I_MS_LR,ratio);
end

%% Measuring data registration
output = dftregistration(fft2(mean(I_GT,3)),fft2(I_PAN),100);
output(3:4)

output = dftregistration(fft2(mean(I_MS,3)),fft2(I_PAN),100);
output(3:4)

output = dftregistration(fft2(mean(I_GT,3)),fft2(mean(I_MS,3)),100);
output(3:4)

%% Inspection RR data

%%% GT
if size(I_MS,3) == 4   
    showImage4(I_GT,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L);    
else
    showImage8(I_GT,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L);
end

%%% MS
if size(I_MS,3) == 4   
    showImage4(I_MS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L);    
else
    showImage8(I_MS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L);
end

%%% PAN
showPan(I_PAN,printEPS,2,flag_cut_bounds,dim_cut);

%% Save results
filename = sprintf('%s_RR.mat',name);
save(filename, 'I_MS', 'I_MS_LR', 'I_GT', 'I_PAN', 'ratio', 'sensor', 'tag_interp', 'bicubic');