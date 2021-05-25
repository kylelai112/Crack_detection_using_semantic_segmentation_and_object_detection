data = load('Gtruth.mat');
crackDataset = data.gTruth.LabelData;
data.gTruth.DataSource.Source;
crackDataset(1:4,:)

allBoxes = vertcat(crackDataset.Crack{:});
aspectRatio = allBoxes(:,3) ./ allBoxes(:,4);
area = prod(allBoxes(:,3:4),2);

figure
scatter(area,aspectRatio)
xlabel("Box Area")
ylabel("Aspect Ratio (width/height)");
title("Box Area vs. Aspect Ratio")

trainingData = boxLabelDatastore(crackDataset(:,:));

numAnchors = 5;
[anchorBoxes,meanIoU] = estimateAnchorBoxes(trainingData,numAnchors);

maxNumAnchors = 15;
meanIoU = zeros([maxNumAnchors,1]);
anchorBoxes = cell(maxNumAnchors, 1);
for k = 1:maxNumAnchors
    % Estimate anchors and mean IoU.
    [anchorBoxes{k},meanIoU(k)] = estimateAnchorBoxes(trainingData,k);    
end

figure
plot(1:maxNumAnchors,meanIoU,'-o')
ylabel("Mean IoU")
xlabel("Number of Anchors")
title("Number of Anchors vs. Mean IoU")

