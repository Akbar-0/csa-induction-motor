function Xaug = augmentSignals(X,augmentFactor)
%   augmentSignals Apply simple augmentations to 1D signals
%   X is [L 1 C N] numeric (C channels). augmentFactor is how many
%   augmented copies to produce for each sample (default 1 -> one extra).

if nargin<2 || isempty(augmentFactor)
    augmentFactor = 1;
end
[L,~,C,N] = size(X);
Xaug = zeros(L,1,C,N*augmentFactor);
idx = 1;
for n=1:N
    x = squeeze(X(:,:, :, n)); % [L C]
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
        % store
        Xaug(:,:,1:C,idx) = reshape(x1,[L 1 C]);
        idx = idx + 1;
    end
end
end
