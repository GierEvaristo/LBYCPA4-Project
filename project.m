load("net.mat","net");
load("testImds.mat","testImds");

YPred = classify(net,validationImds);
YValidation = validationImds.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);