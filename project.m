% trainingPath = fullfile('shoeTypeClassifierDataset','training');
% validationPath = fullfile('shoeTypeClassifierDataset','validation');
% trainingImds = imageDatastore(trainingPath,'IncludeSubfolders',true,'LabelSource','foldernames');
% validationImds = imageDatastore(validationPath,'IncludeSubfolders',true,'LabelSource','foldernames');


imdsPath = fullfile('ut-zap50k-images-square');
imds = imageDatastore(imdsPath,'IncludeSubfolders',true,'LabelSource','foldernames');

% Specify the ratios for the train-test-validation split (e.g., 80% train)
trainRatio = 0.8;

% Perform the train-test split
[trainingImds, validationImds] = splitEachLabel(imds, trainRatio);


augmentation = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...  % Random rotation between -10 and 10 degrees
    'RandXTranslation', [-30, 30], ...  % Random horizontal translation between -30 and 30 pixels
    'RandYTranslation', [-30, 30], ...  % Random vertical translation between -30 and 30 pixels
    'RandXScale', [0.8, 1.2], ...  % Random horizontal scaling between 0.8 and 1.2
    'RandYScale', [0.8, 1.2], ...  % Random vertical scaling between 0.8 and 1.2
    'RandXShear', [-20, 20], ...  % Random horizontal shear between -20 and 20 degrees
    'RandYShear', [-20, 20] ...  % Random vertical shear between -20 and 20 degrees
);

trainingImds = augmentedImageDatastore([136 136],trainingImds,'ColorPreprocessing','gray2rgb','DataAugmentation', augmentation);
validationImds = augmentedImageDatastore([136 136],validationImds,'ColorPreprocessing','gray2rgb','DataAugmentation', augmentation);

figure;
perm = randperm(1494,20);
for i = 1:20
    subplot(4,5,i);
    imshow(trainingImds.Files{perm(i)});
end

layers = [
    imageInputLayer([136 136 3]) % Input layer for a 136x136 RGB image
    
    convolution2dLayer(11, 96, 'Stride', 4, 'Padding', 0) % Convolutional layer 1
    reluLayer % ReLU activation layer
    crossChannelNormalizationLayer(5) % Cross-channel normalization layer
    maxPooling2dLayer(3, 'Stride', 2) % Max pooling layer
    
    convolution2dLayer(5, 256, 'Stride', 1, 'Padding', 2) % Convolutional layer 2
    reluLayer % ReLU activation layer
    crossChannelNormalizationLayer(5) % Cross-channel normalization layer
    maxPooling2dLayer(3, 'Stride', 2) % Max pooling layer
    
    convolution2dLayer(3, 384, 'Stride', 1, 'Padding', 1) % Convolutional layer 3
    reluLayer % ReLU activation layer
    
    convolution2dLayer(3, 384, 'Stride', 1, 'Padding', 1) % Convolutional layer 4
    reluLayer % ReLU activation layer
    
    convolution2dLayer(3, 256, 'Stride', 1, 'Padding', 1) % Convolutional layer 5
    reluLayer % ReLU activation layer
    maxPooling2dLayer(3, 'Stride', 2) % Max pooling layer
    
    fullyConnectedLayer(4096) % Fully connected layer 1
    reluLayer % ReLU activation layer
    dropoutLayer(0.5) % Dropout layer for regularization
    
    fullyConnectedLayer(4096) % Fully connected layer 2
    reluLayer % ReLU activation layer
    dropoutLayer(0.5) % Dropout layer for regularization
    
    fullyConnectedLayer(4) % Fully connected layer 3 (output layer)
    softmaxLayer % Softmax activation layer
    classificationLayer % Classification layer
];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',50, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationImds, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu');

net = trainNetwork(trainingImds,layers,options);

YPred = classify(net,validationImds);
YValidation = validationImds.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);