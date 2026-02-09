%% ===================== SEGMENT-LEVEL K-FOLD (SVM + Bagged Trees) =====================
% This script performs 10-fold cross-validation on EEG segment-level features
% extracted using the paper-faithful feature extraction pipeline. It trains
% and evaluates two classifiers: Bagged Trees (ensemble of decision trees)
% and Support Vector Machine (SVM) with an RBF kernel. Performance metrics
% (accuracy, sensitivity, specificity, F1-score) are computed for each band
% (All, Alpha, Beta), and ROC curves are plotted and saved.

clc; clear; close all;

%% ---------------- CONFIG ----------------
csvFile = 'C:\Users\asafd\Downloads\OneDrive_2025-12-10\PD REST\Final_Features_PaperFaithful_IC_NO_APPROX.csv';
bands    = {'All','Alpha','Beta'};
bandCode = [1 2 3];
K = 10;

rng(1,'twister');

outDir = 'C:\Users\asafd\OneDrive\שולחן העבודה\שנה ד סמסטר א\עיבוד אותות רפואיים\Matlab';
if ~exist(outDir,'dir'), mkdir(outDir); end

%% ---------------- LOAD ----------------
T = readtable(csvFile);
vn = T.Properties.VariableNames;

bandCol = find(strcmpi(vn,'BandType'),1);
subjCol = find(strcmpi(vn,'SubjectID'),1);

if isempty(bandCol) || isempty(subjCol)
    error('BandType or SubjectID column missing');
end

%% ---------------- LABELS (AS BEFORE) ----------------
labels01 = double(T{:,subjCol} < 900);     % HC=0 , PD=1
Yall = categorical(labels01, [0 1], {'HC','PD'});

if numel(categories(removecats(Yall))) < 2
    error('Label rule produced only one class overall. Fix mapping.');
end

%% -------- Summary collectors (Sen/Spe per band) --------
Summary = table('Size',[0 7], ...
    'VariableTypes',{'string','double','double','double','double','double','double'}, ...
    'VariableNames',{'Band','SVM_Sen','SVM_Spe','SVM_Acc','Bag_Sen','Bag_Spe','Bag_Acc'});

%% ===================== MAIN LOOP =====================
for b = 1:3

    fprintf('\n============================\n');
    fprintf('Band: %s (BandType=%d)\n', bands{b}, bandCode(b));
    fprintf('============================\n');

    idx = (T{:,bandCol} == bandCode(b));
    X0  = table2array(T(idx,5:end));
    Y   = Yall(idx);

    fprintf('N=%d | PD=%d | HC=%d | Features=%d\n', ...
        numel(Y), sum(Y=='PD'), sum(Y=='HC'), size(X0,2));

    %% -------- Feature cleaning (same as you had) --------
    good = all(isfinite(X0),1);
    X = X0(:,good);

    v = var(X,0,1);
    X = X(:,v>1e-6);

    C = corr(X);
    rm = false(1,size(C,1));
    for i=1:size(C,1)
        for j=i+1:size(C,1)
            if abs(C(i,j))>0.95
                rm(j)=true;
            end
        end
    end
    X = X(:,~rm);

    fprintf('Features after cleaning: %d\n', size(X,2));

    %% ================= BAGGED TREES (KFold on segments) =================
    tTree = templateTree('MaxNumSplits',20,'MinLeafSize',5);

    mdlBag = fitcensemble(X,Y,'Method','Bag', ...
        'NumLearningCycles',100,'Learners',tTree);

    cvBag = crossval(mdlBag,'KFold',K);
    [predBag, scoreBag] = kfoldPredict(cvBag);

    metBag = compute_metrics(Y,predBag,'PD');
    fprintf('[Bagged] Acc=%.2f%% | Sen=%.2f%% | Spe=%.2f%% | F1=%.2f%%\n', ...
        100*metBag.Accuracy, 100*metBag.Sensitivity, 100*metBag.Specificity, 100*metBag.F1);

    %% ================= SVM (KFold on segments) =================
    mdlSVM = fitcsvm(X,Y,'KernelFunction','rbf', ...
        'KernelScale','auto','Standardize',true, ...
        'ClassNames',categorical({'HC','PD'}));

    mdlSVM = fitPosterior(mdlSVM);
    cvSVM  = crossval(mdlSVM,'KFold',K);
    [predSVM, scoreSVM] = kfoldPredict(cvSVM);

    metSVM = compute_metrics(Y,predSVM,'PD');
    fprintf('[SVM]   Acc=%.2f%% | Sen=%.2f%% | Spe=%.2f%% | F1=%.2f%%\n', ...
        100*metSVM.Accuracy, 100*metSVM.Sensitivity, 100*metSVM.Specificity, 100*metSVM.F1);

    %% -------- Save into summary table --------
    Summary = [Summary; {string(bands{b}), ...
        metSVM.Sensitivity, metSVM.Specificity, metSVM.Accuracy, ...
        metBag.Sensitivity, metBag.Specificity, metBag.Accuracy}];

    %% ================= ROC (Smoothed) =================
    ybin = double(Y=='PD');

    % SVM ROC
    posIdxS = find(string(cvSVM.ClassNames)=="PD",1);
    sS = scoreSVM(:,posIdxS);
    [Xs, Ys, ~, aucS] = perfcurve(ybin, sS, 1);
    [xqs, yqs] = smooth_roc(Xs, Ys);

    % Bagged ROC (if scores exist)
    hasBagScore = ~isempty(scoreBag) && size(scoreBag,2) >= 2;
    if hasBagScore
        posIdxB = find(string(cvBag.Trained{1}.ClassNames)=="PD",1);
        if isempty(posIdxB), posIdxB = 2; end
        sB = scoreBag(:,posIdxB);
        [Xb, Yb, ~, aucB] = perfcurve(ybin, sB, 1);
        [xqb, yqb] = smooth_roc(Xb, Yb);
    end

    figure('Name',['ROC ',bands{b}]); clf;

    plot(Xs, Ys, '.', 'MarkerSize', 6); hold on;
    plot(xqs, yqs, 'LineWidth', 1.8);
    leg = {'SVM raw','SVM smooth'};

    if hasBagScore
        plot(Xb, Yb, '.', 'MarkerSize', 6);
        plot(xqb, yqb, 'LineWidth', 1.8);
        leg = [leg, {'Bagged raw','Bagged smooth'}];
        title(sprintf('%s ROC | AUC SVM=%.3f | AUC Bag=%.3f', bands{b}, aucS, aucB));
    else
        title(sprintf('%s ROC | AUC SVM=%.3f', bands{b}, aucS));
    end

    grid on; xlabel('FPR'); ylabel('TPR');
    legend(leg,'Location','southeast');

    %% ----- SAVE ROC FIGURE (OVERWRITE) -----
    fnameBase = sprintf('ROC_SEGMENTK_%s', bands{b});
    fnameBase = regexprep(fnameBase,'[^a-zA-Z0-9_\-\.]','_');

    exportgraphics(gcf, fullfile(outDir, [fnameBase '.png']), 'Resolution', 300);
    savefig(gcf, fullfile(outDir, [fnameBase '.fig']));
end

%% -------- Print summary like paper table --------
fprintf('\n================ SUMMARY (Sen/Spe/Acc) ================\n');
disp(Summary);

% Optional: save to CSV (overwrite)
writetable(Summary, fullfile(outDir,'Summary_Sen_Spe_Acc.csv'));

fprintf('\n*** DONE (Segment-level KFold) ***\n');

%% ===================== LOCAL FUNCTIONS =====================
function met = compute_metrics(yTrue, yPred, posLabel)
    yTrue = categorical(yTrue);
    yPred = categorical(yPred);

    pos = categorical({posLabel});

    tp = sum(yTrue==pos & yPred==pos);
    tn = sum(yTrue~=pos & yPred~=pos);
    fp = sum(yTrue~=pos & yPred==pos);
    fn = sum(yTrue==pos & yPred~=pos);

    met.Accuracy    = (tp+tn) / max(tp+tn+fp+fn,1);
    met.Precision   = tp / max(tp+fp,1);
    met.Recall      = tp / max(tp+fn,1);
    met.Sensitivity = met.Recall;
    met.Specificity = tn / max(tn+fp,1);
    met.F1          = 2*(met.Precision*met.Recall) / max(met.Precision+met.Recall, eps);
end

function [xq, yq] = smooth_roc(Xroc, Yroc)
    [Xu, ia] = unique(Xroc, 'stable');
    Yu = Yroc(ia);

    xq = linspace(0,1,2000);
    yq = interp1(Xu, Yu, xq, 'pchip', 'extrap');
    yq = min(max(yq,0),1);
end