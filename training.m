imdsPath = fullfile('ut-zap50k-images-square');
imds = imageDatastore(imdsPath,'IncludeSubfolders',true,'LabelSource','foldernames');

% Perform the train-test-validation split
[trainingImds, validationImds] = splitEachLabel(imds, 0.8);
[validationImds, testImds] = splitEachLabel(validationImds,0.5);

augmentation = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-30, 30], ...
    'RandYTranslation', [-30, 30], ...
    'RandXScale', [0.8, 1.2], ...
    'RandYScale', [0.8, 1.2], ...
    'RandXShear', [-20, 20], ...
    'RandYShear', [-20, 20] ...
);

trainingImds = augmentedImageDatastore([136 136],trainingImds,'ColorPreprocessing','gray2rgb','DataAugmentation', augmentation);

figure;
perm = randperm(1494,20);
for i = 1:20
    subplot(4,5,i);
    imshow(trainingImds.Files{perm(i)});
end

layers = [
    imageInputLayer([136 136 3]) % Input layer for a 136x136 RGB image
    
    convolution2dLayer(11, 96, 'Stride', 4, 'Padding', 0)
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer(3, 'Stride', 2)
    
    convolution2dLayer(5, 256, 'Stride', 1, 'Padding', 2)
    reluLayer
    crossChannelNormalizationLayer(5)
    maxPooling2dLayer(3, 'Stride', 2)
    
    convolution2dLayer(3, 384, 'Stride', 1, 'Padding', 1)
    reluLayer
    
    convolution2dLayer(3, 384, 'Stride', 1, 'Padding', 1)
    reluLayer
    
    convolution2dLayer(3, 256, 'Stride', 1, 'Padding', 1)
    reluLayer
    maxPooling2dLayer(3, 'Stride', 2)
    
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationImds, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu');

net = trainNetwork(trainingImds,layers,options);
save("net.mat","net");
save("testImds.mat","testImds");