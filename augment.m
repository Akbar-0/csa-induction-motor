function Xaug = augmentSignals(X,augmentFactor)
%AUGMENTSIGNALS Apply simple augmentations to 1D signals
%   X is [L 1 1 N] numeric. augmentFactor is how many augmented copies
%   to produce for each sample (default 1 -> produce one extra per sample).

if nargin<2 || isempty(augmentFactor)
    augmentFactor = 1;
end
[L,~,~,N] = size(X);
Xaug = zeros(L,1,1,N*augmentFactor);
idx = 1;
for n=1:N
    x = squeeze(X(:,:,1,n));
    for k=1:augmentFactor
        % additive gaussian noise
        snrFactor = 0.05 + 0.1*rand;
        x1 = x + snrFactor*randn(size(x));
        % scaling
        scale = 0.85 + 0.3*rand;
        x1 = x1 * scale;
        % small shift (circular)
        shift = round((-0.02 + 0.04*rand)*L);
        x1 = circshift(x1,shift);
        Xaug(:,:,1,idx) = x1;
        idx = idx + 1;
    end
end
end
