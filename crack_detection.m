

wallds = imageDatastore("Crack_detection_samples","IncludeSubfolders",...
    true,"LabelSource","foldernames");


[trainImgs,testImgs] = splitEachLabel(wallds,0.6);

numClasses = numel(categories(wallds.Labels));

net = googlenet;
lgraph = layerGraph(net);

newFc = fullyConnectedLayer(2,"Name","new_fc");
lgraph = replaceLayer(lgraph,"loss3-classifier",newFc);
newOut = classificationLayer("Name","new_out");
lgraph = replaceLayer(lgraph,"output",newOut);


options = trainingOptions("sgdm","InitialLearnRate", 0.001);

testLabels = testImgs.Labels;
inputSize=[224 224];
trainImgs = augmentedImageDatastore(inputSize, trainImgs);
testImgs = augmentedImageDatastore(inputSize, testImgs);

[wallnet,info] = trainNetwork(trainImgs, lgraph, options);

testpreds = classify(wallnet,testImgs);



disp(nnz(testpreds == testLabels)/ numel(testpreds));

confusionchart(testLabels,testpreds);

%% Check result
testds = imageDatastore("Test","IncludeSubfolders",...
    true,"LabelSource","foldernames");
tLabels = testds.Labels;
tImgs = augmentedImageDatastore(inputSize, testds);
tpreds = classify(wallnet,tImgs);

disp(nnz(tpreds == tLabels)/ numel(tpreds));

confusionchart(tLabels,tpreds);


