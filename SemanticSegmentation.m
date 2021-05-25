data = load('gTruthpixel.mat');
imageDir = fullfile(data.gTruth.DataSource.Source);
labelDir = fullfile(data.gTruth.LabelData.PixelLabelData);

imds = imageDatastore(imageDir);
imds.ReadFcn = @customReadDatastoreImage;
classNames = [data.gTruth.LabelDefinitions.Name];
labelIDs   = [data.gTruth.LabelDefinitions.PixelLabelID];

pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
pxds.ReadFcn = @customReadDatastoreImage;
ds = pixelLabelImageDatastore(imds,pxds);


imageSize = [224 224 3];
numClasses = 2;

lgraph = unetLayers(imageSize, numClasses);  % Unet model
% lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");% ResNet-18 model


options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'VerboseFrequency',10);

net = trainNetwork(ds,lgraph,options);


I = imread('Test\Positive\00121.jpg');
I = imresize(I,imageSize(1:2));
imshow(I)
C = semanticseg(I, net);
C = C == 'Crack';
B = labeloverlay(I,C);
figure
montage({I,B});





function data = customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = imresize(data,[224 224]);
end
