function net = trainModel(varargin)
%   trainModel Train 1D-CNN on CSV-based dataset (Deep Learning Toolbox)
%   net = trainModel('dataDir',..., 'inputLength',2000, 'epochs',20)

p = inputParser;
addParameter(p,'dataDir','',@ischar);
addParameter(p,'inputLength',2000,@isnumeric);
addParameter(p,'epochs',20,@isnumeric);
addParameter(p,'augmentFactor',1,@isnumeric);
addParameter(p,'miniBatchSize',64,@isnumeric);
addParameter(p,'learnRate',1e-3,@isnumeric);
addParameter(p,'valFreq',20,@isnumeric);
parse(p,varargin{:});
opts = p.Results;

% Default to data/simulink if no dataDir provided
if isempty(opts.dataDir)
    here = fileparts(mfilename('fullpath'));
    opts.dataDir = fullfile(here,'..','data','simulink');
end

[XTrain,YTrain,XVal,YVal] = loadDataset(opts.dataDir,opts.inputLength,opts.augmentFactor);

classes = categories(YTrain);
numClasses = numel(classes);

% Determine number of channels from loaded data
numChannels = size(XTrain,3);

layers = create1DCNN(opts.inputLength,numClasses,numChannels);

% training options (CPU)
trainingOpts = trainingOptions('adam', ...
    'InitialLearnRate',opts.learnRate, ...
    'MaxEpochs',opts.epochs, ...
    'MiniBatchSize',opts.miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal,YVal}, ...
    'ValidationFrequency',opts.valFreq, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'ExecutionEnvironment','cpu');

net = trainNetwork(XTrain,YTrain,layers,trainingOpts);
save('trained1dcnn.mat','net');

% evaluate
evalMetrics = evaluateNetwork(net,XVal,YVal);
disp(evalMetrics);
end
