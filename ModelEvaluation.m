yolo2net= load('yolov2Model2.mat');
unet= load('unetSegModel.mat');
rs18net= load('resnet18SegModel.mat');
% analyzeNetwork(net);

imageSize = [224 224 3];

%Read different images

% I = imread('Crack_detection_samples\Positive\00100.jpg');
I = imread('Test\Positive\00262.jpg');
% I = imread('Test\Positive\00259.jpg');
% I = imread('Test\Positive\00206.jpg');
% I = imread('Test\Positive\00252.jpg');
% I = imread('Test\Positive\00278.jpg');
% I = imread('Test\Positive\00263.jpg');
% I = imread('Test\Positive\00251.jpg');
% I = imread('Test\Positive\00240.jpg');
% I = imread('Test\Positive\00236.jpg');
% I = imread('Test\Positive\00128.jpg');

I = imresize(I,imageSize(1:2));


%YOLOv2
[bboxes,scores] = detect(yolo2net.detector,I);

% unet
C = semanticseg(I, unet.net);
C = C == 'Crack';
B = labeloverlay(I,C);

%rs18net
D = semanticseg(I, rs18net.net);
D = D == 'Crack';
E = labeloverlay(I,D);

if  isempty(bboxes)
    F = imread('Test\Positive\fail.png');
else
    F = insertObjectAnnotation(I,'rectangle',bboxes,scores);
end
    
figure
subplot(141)
imshow(I), title('Original');
subplot(142)
imshow(F), title('YOLO v2');
subplot(143)
imshow(B), title('Unet');
subplot(144)
imshow(E), title('ResNet-18');