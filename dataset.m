function [XTrain,YTrain,XVal,YVal] = loadDataset(dataDir,inputLength,augmentFactor)
%LOADDATASET Load dataset or synthesize data if none found.
%   [XTrain,YTrain,XVal,YVal] = LOaddATASET(dataDir,inputLength,augmentFactor)

if nargin<2 || isempty(inputLength)
    inputLength = 2000;
end
if nargin<3
    augmentFactor = 1;
end

if nargin<1 || isempty(dataDir) || ~isfolder(dataDir)
    warning('No data folder found; generating synthetic data for testing.');
    [X,Y] = generate_synthetic_data(120,inputLength);
else
    % Placeholder: expect .mat files or wav-like signals in dataDir
    % User should replace this block with actual data loading.
    files = dir(fullfile(dataDir,'*.mat'));
    if isempty(files)
        warning('No .mat files found in %s; using synthetic data.',dataDir);
        [X,Y] = generate_synthetic_data(120,inputLength);
    else
        % naive loader: expects variable 'signal' and 'label' in each .mat
        X = [];
        Y = categorical();
        for i=1:numel(files)
            d = load(fullfile(dataDir,files(i).name));
            if isfield(d,'signal') && isfield(d,'label')
                s = d.signal(:)';
                if numel(s) < inputLength
                    s = [s zeros(1,inputLength-numel(s))];
                elseif numel(s) > inputLength
                    s = s(1:inputLength);
                end
                X(:,:,1,end+1) = s(:); %#ok<AGROW>
                Y(end+1,1) = categorical({d.label}); %#ok<AGROW>
            end
        end
    end
end

% split train/val
N = size(X,4);
perm = randperm(N);
numTrain = round(0.8*N);
trainIdx = perm(1:numTrain);
valIdx = perm(numTrain+1:end);
XTrain = X(:,:,:,trainIdx);
YTrain = Y(trainIdx);
XVal = X(:,:,:,valIdx);
YVal = Y(valIdx);

% augmentation for small datasets
if augmentFactor>0
    Xaug = augmentSignals(XTrain,augmentFactor);
    % replicate labels
    Yaug = repmat(YTrain,augmentFactor,1);
    XTrain = cat(4,XTrain,Xaug);
    YTrain = [YTrain; Yaug];
end
end
