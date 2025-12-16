function layers = create1DCNN(inputLength,numClasses,numChannels)
%   create1DCNN returns a 1D-CNN defined using 2D layers (time × 1).
%   layers = create1DCNN(inputLength,numClasses,numChannels) returns an
%   array of layers suitable for training with trainNetwork on time-series
%   of length inputLength and numChannels channels.

if nargin<3 || isempty(numChannels)
    numChannels = 1;
end

layers = [
    imageInputLayer([inputLength 1 numChannels],'Normalization','zscore','Name','input')

    convolution2dLayer([9 1],16,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer([2 1],'Stride',[2 1],'Name','pool1')

    convolution2dLayer([7 1],32,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer([2 1],'Stride',[2 1],'Name','pool2')

    convolution2dLayer([5 1],64,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')

    globalAveragePooling2dLayer('Name','gap')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];
end
