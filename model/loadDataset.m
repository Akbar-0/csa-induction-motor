function [XTrain,YTrain,XVal,YVal] = loadDataset(dataDir,inputLength,augmentFactor)
%LOADDATASET Load dataset from CSVs (preferred) or .mat, else synthesize.
%   [XTrain,YTrain,XVal,YVal] = loadDataset(dataDir,inputLength,augmentFactor)

if nargin<2 || isempty(inputLength)
    inputLength = 2000;
end
if nargin<3
    augmentFactor = 1;
end

% Resolve dataDir default to project CSV path if empty
if nargin<1 || isempty(dataDir)
    % Try default relative path 'data/simulink'
    here = fileparts(mfilename('fullpath'));
    dataDir = fullfile(here,'..','data','simulink');
end

if ~isfolder(dataDir)
    warning('Data folder %s not found; generating synthetic data for testing.',dataDir);
    [X,Y] = generate_synthetic_data(120,inputLength);
else
    % 1) Prefer CSV files in the folder
    csvFiles = dir(fullfile(dataDir,'*.csv'));
    if ~isempty(csvFiles)
        [X,Y] = loadFromCsvFiles(dataDir,csvFiles,inputLength);
    else
        % 2) Fallback to legacy .mat loader (expects variables: signal, label)
        matFiles = dir(fullfile(dataDir,'*.mat'));
        if isempty(matFiles)
            warning('No CSV or .mat files found in %s; generating synthetic data.',dataDir);
            [X,Y] = generate_synthetic_data(120,inputLength);
        else
            X = [];
            Y = categorical();
            for j=1:numel(matFiles)
                d = load(fullfile(dataDir,matFiles(j).name));
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
end

% stratified split per class (80/20) with balanced validation and oversampled training
classes = categories(Y);
trainIdx = [];
valIdx = [];
for ci = 1:numel(classes)
    cidx = find(Y == classes{ci});
    cperm = cidx(randperm(numel(cidx)));
    numTrainC = max(1,round(0.8*numel(cidx))); % ensure at least 1
    trainIdx = [trainIdx; cperm(1:numTrainC)]; %#ok<AGROW>
    if numel(cperm) > numTrainC
        valIdx = [valIdx; cperm(numTrainC+1:end)]; %#ok<AGROW>
    end
end

% Balance validation set to min class count
valIdxBalanced = [];
minValCount = inf;
valPerClass = cell(numel(classes),1);
for ci = 1:numel(classes)
    cidx = find(Y(valIdx) == classes{ci});
    valPerClass{ci} = valIdx(cidx);
    minValCount = min(minValCount, numel(cidx));
end
if isfinite(minValCount) && minValCount>0
    for ci = 1:numel(classes)
        csel = valPerClass{ci};
        if numel(csel) > minValCount
            csel = csel(randperm(numel(csel), minValCount));
        end
        valIdxBalanced = [valIdxBalanced; csel]; %#ok<AGROW>
    end
    valIdx = valIdxBalanced;
end

% Oversample training set to max class count
trainPerClass = cell(numel(classes),1);
maxTrainCount = 0;
for ci = 1:numel(classes)
    cidx = find(Y(trainIdx) == classes{ci});
    trainPerClass{ci} = trainIdx(cidx);
    maxTrainCount = max(maxTrainCount, numel(cidx));
end
trainIdxBalanced = [];
for ci = 1:numel(classes)
    csel = trainPerClass{ci};
    if isempty(csel)
        continue;
    end
    reps = ceil(maxTrainCount / numel(csel));
    caug = repmat(csel, reps, 1);
    caug = caug(1:maxTrainCount);
    trainIdxBalanced = [trainIdxBalanced; caug]; %#ok<AGROW>
end
trainIdx = trainIdxBalanced;

% shuffle combined indices
trainIdx = trainIdx(randperm(numel(trainIdx)));
valIdx = valIdx(randperm(numel(valIdx)));
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

% --- helpers ---
function [X,Y] = loadFromCsvFiles(dataDir,csvFiles,inputLength)
% Load CSVs produced by Simulink with columns: Time,Torque,RPM,Ia,Ib,Ic
% Creates single-channel magnitude current signal sqrt(Ia^2+Ib^2+Ic^2)

% Preallocate for 5-channel inputs [Ia Ib Ic Torque RPM] when available
X = zeros(inputLength,1,5,0);
Y = categorical();
for i=1:numel(csvFiles)
    fpath = fullfile(dataDir,csvFiles(i).name);
    try
        T = readtable(fpath);
    catch
        % fallback: readmatrix if header issues
        M = readmatrix(fpath);
        % assume 6 columns [Time Ia Ib Ic Torque RPM] as per format
        if size(M,2) >= 6
            Ia = M(:,2); Ib = M(:,3); Ic = M(:,4);
            Torque = M(:,5); RPM = M(:,6);
        else
            % if fewer columns, take the last column as signal
            Ia = M(:,end); Ib = zeros(size(Ia)); Ic = zeros(size(Ia));
            Torque = zeros(size(Ia)); RPM = zeros(size(Ia));
        end
        chans = computeChannels(Ia,Ib,Ic,Torque,RPM); % [L 5]
        [X,Y] = appendWindows(X,Y,chans,csvFiles(i).name,inputLength);
        continue;
    end

    % Determine signal columns robustly
    vars = lower(string(T.Properties.VariableNames));
    iaIdx = find(vars=="ia",1);
    ibIdx = find(vars=="ib",1);
    icIdx = find(vars=="ic",1);
    tqIdx = find(vars=="torque",1);
    rpmIdx = find(vars=="rpm",1);
    if ~isempty(iaIdx) && ~isempty(ibIdx) && ~isempty(icIdx)
        Ia = T{:,iaIdx}; Ib = T{:,ibIdx}; Ic = T{:,icIdx};
        if ~isempty(tqIdx) && ~isempty(rpmIdx)
            Torque = T{:,tqIdx}; RPM = T{:,rpmIdx};
        else
            Torque = zeros(size(Ia)); RPM = zeros(size(Ia));
        end
        chans = computeChannels(Ia,Ib,Ic,Torque,RPM);
    else
        % If Ia/Ib/Ic not present, take the last numeric column as signal
        numCols = varfun(@isnumeric,T,'OutputFormat','uniform');
        if any(numCols)
            sig = T{:,find(numCols,1,'last')};
        else
            warning('No numeric columns found in %s. Skipping.',fpath);
            continue;
        end
    end

    if exist('chans','var') && ~isempty(chans)
        [X,Y] = appendWindows(X,Y,chans,csvFiles(i).name,inputLength);
    else
        % Single-channel fallback if only one numeric column is available
        [X,Y] = appendWindows(X,Y,sig(:),csvFiles(i).name,inputLength);
    end
end

if isempty(Y)
    warning('No valid CSV samples could be loaded from %s.',dataDir);
end
end

function chans = computeChannels(Ia,Ib,Ic,Torque,RPM)
Ia = Ia(:); Ib = Ib(:); Ic = Ic(:);
Torque = Torque(:); RPM = RPM(:);
chans = [Ia Ib Ic Torque RPM]; % [L 5]
end

function [X,Y] = appendWindows(X,Y,data,filename,inputLength)
% Slice the data into overlapping windows; data can be [L] or [L 3]
if isvector(data)
    data = data(:); % [L 1]
end
L = size(data,1);
C = size(data,2);
stride = max(1,round(inputLength/2)); % 50% overlap by default
if L < inputLength
    data = [data; zeros(inputLength-L,C)];
    L = inputLength;
end
starts = 1:stride:(L-inputLength+1);
if isempty(starts)
    starts = 1;
end
label = inferLabelFromName(filename);
for windowStart = starts
    w = data(windowStart:windowStart+inputLength-1, :); % [inputLength C]
    if isempty(X)
        X = zeros(inputLength,1,size(w,2),0);
    end
    X(:,1,1:C,end+1) = reshape(w,[inputLength 1 C]); %#ok<AGROW>
    Y(end+1,1) = categorical({label}); %#ok<AGROW>
end
end

function label = inferLabelFromName(fname)
% Map filename patterns to class labels
name = lower(string(fname));
if contains(name,"healthy")
    label = 'Healthy';
elseif contains(name,"bearingfault")
    label = 'BearingFault';
elseif contains(name,"rotorfault")
    label = 'RotorFault';
elseif contains(name,"phaseimbalance")
    label = 'PhaseImbalance';
else
    % default to Unknown; still include it as its own class
    label = 'Unknown';
end
end
end
