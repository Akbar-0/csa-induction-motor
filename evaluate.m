function metrics = evaluateNetwork(net,XTest,YTest)
%EVALUATENETWORK Compute accuracy, precision, recall, and confusion matrix
%   metrics = EVALUATENETWORK(net,XTest,YTest)

YPred = classify(net,XTest);
acc = mean(YPred == YTest);
cm = confusionmat(YTest,YPred);

% compute per-class precision & recall
classes = categories(YTest);
nC = numel(classes);
precision = zeros(nC,1);
recall = zeros(nC,1);
for i=1:nC
    tp = cm(i,i);
    fp = sum(cm(:,i)) - tp;
    fn = sum(cm(i,:)) - tp;
    if tp+fp==0
        precision(i)=0;
    else
        precision(i) = tp / (tp+fp);
    end
    if tp+fn==0
        recall(i)=0;
    else
        recall(i) = tp / (tp+fn);
    end
end

metrics.Accuracy = acc;
metrics.ConfusionMatrix = cm;
metrics.Classes = classes;
metrics.Precision = precision;
metrics.Recall = recall;

fprintf('Accuracy: %.3f\n',acc);
for i=1:nC
    fprintf('Class %s — Precision %.3f  Recall %.3f\n',string(classes(i)),precision(i),recall(i));
end

figure;
confusionchart(YTest,YPred);
title('Confusion Matrix');
end
