trainingPath = fullfile('shoeTypeClassifierDataset','training');
validationPath = fullfile('shoeTypeClassifierDataset','validation');
trainingImds = imageDatastore(trainingPath,'IncludeSubfolders',true,'LabelSource','foldernames');
validationImds = imageDatastore(validationPath,'IncludeSubfolders',true,'LabelSource','foldernames');

trainingImds = augmentedImageDatastore([200 200],trainingImds,'ColorPreprocessing','gray2rgb');
validationImds = augmentedImageDatastore([200 200],validationImds,'ColorPreprocessing','gray2rgb');

figure;
perm = randperm(1494,20);
for i = 1:20
    subplot(4,5,i);
    imshow(trainingImds.Files{perm(i)});
end

layers = [
    imageInputLayer([200 200 3])

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
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