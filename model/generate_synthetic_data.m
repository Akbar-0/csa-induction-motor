% Deprecated due to usage of Simulink for data generation

function [X,Y] = generate_synthetic_data(numPerClass,inputLength)
%   generate_synthetic_data Create simple synthetic 1D signals for 4 classes
%   [X,Y] = generate_synthetic_data(numPerClass,inputLength) returns
%   X sized [inputLength 1 1 N] and categorical labels Y of length N.

if nargin<1 || isempty(numPerClass)
    numPerClass = 100;
end
if nargin<2 || isempty(inputLength)
    inputLength = 2000;
end

rng(0);
classes = {'Healthy','Bearing','BrokenRotorBar','StatorShort'};
N = numPerClass * numel(classes);
X = zeros(inputLength,1,1,N);
Y = categorical();

t = linspace(0,1,inputLength);
idx = 1;
for c=1:numel(classes)
    for n=1:numPerClass
        switch classes{c}
            case 'Healthy'
                sig = sin(2*pi*50*t) + 0.1*randn(1,inputLength);
            case 'Bearing'
                sig = sin(2*pi*50*t) + 0.5*sin(2*pi*200*t).*exp(-50*(t-0.5).^2) + 0.15*randn(1,inputLength);
            case 'BrokenRotorBar'
                sig = sin(2*pi*50*t) + 0.3*square(2*pi*5*t) + 0.15*randn(1,inputLength);
            case 'StatorShort'
                sig = 0.6*sin(2*pi*120*t) + 0.2*randn(1,inputLength);
        end
        % random scaling and small time-shift
        scale = 0.8 + 0.4*rand;
        shift = round((rand-0.5)*0.05*inputLength);
        s2 = imtranslate(scale*sig,[shift 0],'FillValues',0);
        X(:,:,1,idx) = s2(:);
        Y(idx,1) = categorical({classes{c}});
        idx = idx + 1;
    end
end
end
