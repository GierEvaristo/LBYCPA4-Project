function [accuracy]=project
load("net.mat","net");
load("testImds.mat","testImds");

pred = classify(net,testImds);
test = testImds.Labels;

figure(1)
perm = randperm(1494,20);
for i = 1:20
    subplot(4,5,i);
    index = perm(i);
    imshow(testImds.Files{index});
    title(test(index))
end

accuracy = sum(pred == test)/numel(test);
figure(2)
plotconfusion(test,pred)