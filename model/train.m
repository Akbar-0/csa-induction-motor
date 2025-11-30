function net = trainModel(varargin)
%TRAINMODEL Train 1D-CNN on loaded dataset (MATLAB Deep Learning Toolbox)
%   net = TRAINMODEL('dataDir',..., 'inputLength',2000, 'epochs',20)

p = inputParser;
addParameter(p,'dataDir','',@ischar);
addParameter(p,'inputLength',2000,@isnumeric);
addParameter(p,'epochs',20,@isnumeric);
addParameter(p,'augmentFactor',1,@isnumeric);
addParameter(p,'miniBatchSize',64,@isnumeric);
addParameter(p,'learnRate',1e-3,@isnumeric);
parse(p,varargin{:});
opts = p.Results;

[XTrain,YTrain,XVal,YVal] = loadDataset(opts.dataDir,opts.inputLength,opts.augmentFactor);

classes = categories(YTrain);
numClasses = numel(classes);

layers = create1DCNN(opts.inputLength,numClasses);

% training options (CPU)
trainingOpts = trainingOptions('adam', ...
    'InitialLearnRate',opts.learnRate, ...
    'MaxEpochs',opts.epochs, ...
    'MiniBatchSize',opts.miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal,YVal}, ...
    'ValidationFrequency',30, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'ExecutionEnvironment','cpu');

net = trainNetwork(XTrain,YTrain,layers,trainingOpts);
save('trained1dcnn.mat','net');

% evaluate
evalMetrics = evaluateNetwork(net,XVal,YVal);
disp(evalMetrics);
end
