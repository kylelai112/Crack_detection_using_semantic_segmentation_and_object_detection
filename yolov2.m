data = load('Gtruth.mat');
lblBox = boxLabelDatastore(data.gTruth.LabelData);
imPath = imageDatastore(data.gTruth.DataSource.Source);
crackData = combine(imPath, lblBox);
scaledData = transform(crackData,@scaleGT);

anchorBoxes = estimateAnchorBoxes(scaledData,10);

%% showing one label image example
data = read(crackData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%%
net = resnet18;
numClasses = 1;
imageSize = [224 224 3];

lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,net,...
    "res5b_relu","ReorgLayerSource","res3a_relu");

options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20,...
        'VerboseFrequency',10);

detector = trainYOLOv2ObjectDetector(scaledData,lgraph,options);

I = imread('Crack_detection_samples\Positive\00100.jpg');
I = imresize(I,imageSize(1:2));
[bboxes,scores] = detect(detector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)




function data = scaleGT(data)  
    targetSize = [224 224];
    % data{1} is the image
    scale = targetSize./size(data{1},[1 2]);
    data{1} = imresize(data{1},targetSize);
    % data{2} is the bounding box
    data{2} = bboxresize(data{2},scale);
end