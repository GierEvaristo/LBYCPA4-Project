load("net.mat","net");
load("testImds.mat","testImds");

YPred = classify(net,testImds);
YValidation = testImds.Labels;

figure;
perm = randperm(1494,20);
for i = 1:20
    subplot(4,5,i);
    index = perm(i);
    imshow(testImds.Files{index});
    title(YPred(index))
end

accuracy = sum(YPred == YValidation)/numel(YValidation);