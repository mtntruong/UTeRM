clc;
clear;
close all;

addpath('./ReducedResolutionProtocolToolbox');

ratio = 4;
L = 64;

%sensor = 'WV2'; channels = 8; ms_path = '5_WorldView-2';
%sensor = 'WV3'; channels = 8; ms_path = '6_WorldView-3';
 sensor = 'WV4'; channels = 4; ms_path = '4_WorldView-4';
%sensor = 'QB' ; channels = 4; ms_path = '2_QuickBird';
%sensor = 'IKONOS' ; channels = 4; ms_path = '1_IKONOS';

tag = sensor;
if strcmp(sensor, 'GF-2')
    tag = 'none';
end

% number of images and filenames
num_img = length(dir(fullfile('Satellite-MS', ms_path, 'MS_256'))) - 2;
for i = 1 : num_img
    mat_files{i} = strcat(string(i), '.mat');
end

% split training/testing data
sample_index = 1:num_img;
train_index  = datasample(sample_index, round(num_img * 0.7), 'Replace', false); % 70% for training
test_index   = setdiff(sample_index, train_index); % the rest are for testing
fname1 = sprintf('./%s_train.h5', sensor);
fname2 = sprintf('./%s_test.h5', sensor);

%------ TESTING DATA ------%

% init
PANS  = zeros(256, 256, 1, length(test_index));        % pseudo-hrpan
FPANS = zeros(1024, 1024, 1, length(test_index));      % full-pan
HRMSS = zeros(256, 256, channels, length(test_index)); % pseudo-hrms
USMSS = zeros(256, 256, channels, length(test_index)); % upsampled
LRMSS = zeros(64,  64,  channels, length(test_index)); % lrms

% read MATs
for i = 1 : length(test_index)
    load(fullfile('Satellite-MS', ms_path, 'MS_256',   mat_files{test_index(i)}), 'imgMS')
    load(fullfile('Satellite-MS', ms_path, 'PAN_1024', mat_files{test_index(i)}), 'imgPAN')
    dMS = double(imgMS) / 2047; dPAN = double(imgPAN) / 2047;
    
    [I_MS_LR, I_PAN] = resize_images(dMS, dPAN, 4, tag);

    crop_pan  = I_PAN;
    crop_lrms = I_MS_LR;
    crop_hrms = dMS;
    crop_usms = interp23tap(crop_lrms, 4);
    
    PANS(:, :, 1, i)  = crop_pan;
    FPANS(:, :, 1, i) = dPAN;
    HRMSS(:, :, :, i) = crop_hrms;
    LRMSS(:, :, :, i) = crop_lrms;
    USMSS(:, :, :, i) = crop_usms;
end

% save testing dataset
fname = fname2;
h5create(fname, '/PANS',  size(PANS), 'Deflate',9,'Datatype','single','ChunkSize',[L/2, L/2, 1,          4]);
h5write(fname, '/PANS', single(PANS));
h5create(fname, '/FPANS', size(FPANS),'Deflate',9,'Datatype','single','ChunkSize',[L/2, L/2, 1,          4]);
h5write(fname,'/FPANS', single(FPANS));
h5create(fname, '/LRMSS', size(LRMSS),'Deflate',9,'Datatype','single','ChunkSize',[L/2, L/2, channels/2, 4]);
h5write(fname,'/LRMSS', single(LRMSS));
h5create(fname, '/HRMSS', size(HRMSS),'Deflate',9,'Datatype','single','ChunkSize',[L/2, L/2, channels/2, 4]);
h5write(fname,'/HRMSS', single(HRMSS));
h5create(fname, '/USMSS', size(USMSS),'Deflate',9,'Datatype','single','ChunkSize',[L/2, L/2, channels/2, 4]);
h5write(fname,'/USMSS', single(USMSS));

%------ TRAINING DATA ------%
clear PANS HRMSS USMSS LRMSS

% init
PANS  = zeros(256, 256, 1, length(train_index) * 9);        % pseudo-hrpan
HRMSS = zeros(256, 256, channels, length(train_index) * 9); % pseudo-hrms
USMSS = zeros(256, 256, channels, length(train_index) * 9); % upsampled
LRMSS = zeros(64,  64,  channels, length(train_index) * 9); % lrms

% read MATs
idx = 0;
for i = 1 : length(train_index)
    load(fullfile('Satellite-MS', ms_path, 'MS_256',   mat_files{train_index(i)}), 'imgMS')
    load(fullfile('Satellite-MS', ms_path, 'PAN_1024', mat_files{train_index(i)}), 'imgPAN')
    dMS = double(imgMS) / 2047; dPAN = double(imgPAN) / 2047;
    
    [I_MS_LR, I_PAN] = resize_images(dMS, dPAN, 4, tag);

    crop_pan  = I_PAN;
    crop_lrms = I_MS_LR;
    crop_hrms = dMS;
    crop_usms = interp23tap(crop_lrms, 4);
    
    for j = 1 : 9
        idx = idx + 1;
        
        aug_crop_pan  = PatchAugmentation(crop_pan, j);
        aug_crop_hrms = PatchAugmentation(crop_hrms, j);
        aug_crop_lrms = PatchAugmentation(crop_lrms, j);
        aug_crop_usms = PatchAugmentation(crop_usms, j);
        
        PANS(:, :, 1, idx)  = aug_crop_pan;
        HRMSS(:, :, :, idx) = aug_crop_hrms;
        LRMSS(:, :, :, idx) = aug_crop_lrms;
        USMSS(:, :, :, idx) = aug_crop_usms;
    end
end


% save training dataset
fname = fname1;
h5create(fname, '/PANS',  size(PANS), 'Deflate',9,'Datatype','single','ChunkSize',[L/2, L/2, 1,          4]);
h5write(fname, '/PANS', single(PANS));
h5create(fname, '/LRMSS', size(LRMSS),'Deflate',9,'Datatype','single','ChunkSize',[L/2, L/2, channels/2, 4]);
h5write(fname,'/LRMSS', single(LRMSS));
h5create(fname, '/HRMSS', size(HRMSS),'Deflate',9,'Datatype','single','ChunkSize',[L/2, L/2, channels/2, 4]);
h5write(fname,'/HRMSS', single(HRMSS));
h5create(fname, '/USMSS', size(USMSS),'Deflate',9,'Datatype','single','ChunkSize',[L/2, L/2, channels/2, 4]);
h5write(fname,'/USMSS', single(USMSS));
