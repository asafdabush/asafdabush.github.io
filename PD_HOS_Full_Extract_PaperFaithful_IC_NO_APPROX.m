function PD_HOS_Full_Extract_PaperFaithful_IC_NO_APPROX()
% =========================================================================
% Paper-faithful feature extraction (NO approximations) - ICA components
% Paper: "Diagnosis of Parkinson's disease using higher order statistical
% analysis of alpha and beta rhythms" (BSPC 2022)
%
% Assumptions:
% - Fs = 500 Hz
% - 2 s non-overlapping segments -> 1000 samples
% - up to 20 segments per participant
% - 3 datasets: AllBands, Alpha (8-12), Beta (13-30)
% - Features extracted from ICA component signals (IC1..)
%
% FIXES in this version:
% - Enforce fixed MIN_ICS = 44 components for ALL rows/headers
% - Skip files with < 44 ICs
% - If srate/epoch length mismatch: warn + skip (no full stop)
% =========================================================================

clc; clear; close all;

%% ---------------- CONFIG ----------------
base_folder  = 'C:\Users\asafd\Downloads\OneDrive_2025-12-10\PD REST';

% Folder with .set files
input_folder = fullfile(base_folder, 'Preprocessed_Article_ICA_PaperFaithful');

% Output CSV
output_file  = fullfile(base_folder, 'Final_Features_PaperFaithful_IC.csv');
% Recommended:
% output_file  = fullfile(base_folder, 'Final_Features_PaperFaithful_IC_NO_APPROX.csv');

fs_target = 500;
seg_len   = 1000;
max_segs  = 20;

% Enforce fixed number of ICs
MIN_ICS = 44;

% LLE params (Rosenstein)
lle_m       = 3;
lle_tau     = 10;
lle_twin    = 50;
lle_fit_len = 20;

% CD params (GP)
cd_dim = 3;
cd_tau = 2;

% ApEn params
apen_m = 2;
apen_r = 0.2;

% HOS params
nfft      = 256;
phaseBins = 18;

% Parallel?
hasPCT = license('test','Distrib_Computing_Toolbox');
if hasPCT
    p = gcp('nocreate');
    if isempty(p), parpool; end
end

% EEGLAB
[~,~,~,~] = eeglab; %#ok<ASGLU>

%% -------------- FILE LIST --------------
files = dir(fullfile(input_folder, '*.set'));
if isempty(files)
    error('No .set files found in input folder: %s', input_folder);
end

%% -------------- LOAD FIRST FILE FOR LABELS --------------
EEG0 = pop_loadset('filename', files(1).name, 'filepath', input_folder);

if EEG0.nbchan < MIN_ICS
    error('First file has only %d ICs (<%d). Choose a file with >=%d ICs or lower MIN_ICS.', ...
        EEG0.nbchan, MIN_ICS, MIN_ICS);
end

% Take first MIN_ICS labels only (fixed schema)
if isfield(EEG0.chanlocs,'labels')
    all_labels = {EEG0.chanlocs.labels};
elseif isfield(EEG0.chanlocs,'label')
    all_labels = {EEG0.chanlocs.label};
else
    error('chanlocs has no label(s) field.');
end

chan_labels = all_labels(1:MIN_ICS);

fprintf('Using fixed %d components (IC1..IC%d).\n', MIN_ICS, MIN_ICS);

% Sanity: should look like IC1..ICn
if ~any(startsWith(string(chan_labels), "IC", 'IgnoreCase', true))
    warning('Channel labels do not start with "IC". Continuing anyway, but verify input is IC-based.');
end

%% -------------- HEADERS --------------
feat_names = {'LLE','HE','ApEn','CD', ...
              'Mu3','Mu4','Mu5','Mu6','Mu7','Mu8', ...
              'H1','H2','H3','H4','H5', ...
              'BH1','BH2','BH3','BH4','BH5', ...
              'BE1','BE2','BEp'};

headers = {'SubjectID','Group','BandType','SegmentID'}; % BandType: 1=All,2=Alpha,3=Beta
for f = 1:numel(feat_names)
    for c = 1:MIN_ICS
        headers{end+1} = sprintf('%s_%s', feat_names{f}, sanitize_label(chan_labels{c}));
    end
end

% Initialize CSV (write headers once)
if exist(output_file,'file'), delete(output_file); end
write_csv_header(output_file, headers);

%% -------------- MAIN LOOP --------------
totalRows = 0;
countBand = zeros(1,3);

skipped_name  = 0;
skipped_ics   = 0;
skipped_srate = 0;
skipped_len   = 0;

for i = 1:numel(files)
    fname = files(i).name;

    meta = parse_filename_paper(fname);
    if isnan(meta.SubjectID) || meta.BandType==0 || isnan(meta.Group)
        fprintf('Skipping (unrecognized name/group): %s\n', fname);
        skipped_name = skipped_name + 1;
        continue;
    end

    fprintf('\n[%d/%d] Loading: %s | Subj=%d | Group=%d | Band=%d\n', ...
        i, numel(files), fname, meta.SubjectID, meta.Group, meta.BandType);

    EEG = pop_loadset('filename', fname, 'filepath', input_folder);

    % Enforce fixed IC count
    if EEG.nbchan < MIN_ICS
        warning('Skipping %s: nbchan=%d (<%d required)', fname, EEG.nbchan, MIN_ICS);
        skipped_ics = skipped_ics + 1;
        continue;
    end

    % Enforce paper constraints (DON''T STOP ALL RUN)
    if EEG.srate ~= fs_target
        warning('Skipping %s: srate=%g (expected %d)', fname, EEG.srate, fs_target);
        skipped_srate = skipped_srate + 1;
        continue;
    end

    if size(EEG.data,2) ~= seg_len
        warning('Skipping %s: epoch length=%d (expected %d)', fname, size(EEG.data,2), seg_len);
        skipped_len = skipped_len + 1;
        continue;
    end

    % Use first 20 segments per participant per paper
    if EEG.trials < max_segs
        warning('File %s has only %d trials; paper uses 20. Will use %d.', ...
            fname, EEG.trials, EEG.trials);
        nUse = EEG.trials;
    else
        nUse = max_segs;
    end

    % Preallocate chunk: rows = nUse, cols = 4 + 23*MIN_ICS
    nFeat = 23;
    nCols = 4 + nFeat*MIN_ICS;
    chunk = NaN(nUse, nCols);

    for ep = 1:nUse
        % Take first 44 ICs only
        X = double(EEG.data(1:MIN_ICS,:,ep)); % MIN_ICS x 1000

        % z-score per component
        mu = mean(X,2);
        sd = std(X,0,2);

        % avoid division by ~0
        flat = sd < 1e-12;
        sd(flat) = NaN;

        Xz = (X - mu)./sd;  % flat -> NaN row

        epoch_feats = NaN(nFeat, MIN_ICS);

        if hasPCT
            parfor ch = 1:MIN_ICS
                epoch_feats(:,ch) = compute_all_features_one_channel_NO_APPROX( ...
                    Xz(ch,:), fs_target, ...
                    lle_m, lle_tau, lle_twin, lle_fit_len, ...
                    apen_m, apen_r, ...
                    cd_dim, cd_tau, ...
                    nfft, phaseBins);
            end
        else
            for ch = 1:MIN_ICS
                epoch_feats(:,ch) = compute_all_features_one_channel_NO_APPROX( ...
                    Xz(ch,:), fs_target, ...
                    lle_m, lle_tau, lle_twin, lle_fit_len, ...
                    apen_m, apen_r, ...
                    cd_dim, cd_tau, ...
                    nfft, phaseBins);
            end
        end

        rowvec = reshape(epoch_feats', 1, []);
        chunk(ep,:) = [meta.SubjectID, meta.Group, meta.BandType, ep, rowvec];
    end

    append_numeric_rows_csv(output_file, chunk);

    totalRows = totalRows + nUse;
    countBand(meta.BandType) = countBand(meta.BandType) + nUse;

    fprintf('  -> Appended %d rows. Totals: All=%d Alpha=%d Beta=%d | Total=%d\n', ...
        nUse, countBand(1), countBand(2), countBand(3), totalRows);
end

fprintf('\nDONE.\n');
fprintf('Skipped: name/group=%d | <44IC=%d | srate=%d | epochLen=%d\n', ...
    skipped_name, skipped_ics, skipped_srate, skipped_len);
fprintf('Expected per band ≈ 56*20 = 1120 (if all files exist). Total expected ≈ 3360.\n');
fprintf('Final counts: All=%d Alpha=%d Beta=%d | Total=%d\n', ...
    countBand(1), countBand(2), countBand(3), totalRows);
fprintf('Saved: %s\n', output_file);

end

%% =========================================================================
% CORE FEATURE WRAPPER (NO approximations)
%% =========================================================================
function feats = compute_all_features_one_channel_NO_APPROX(x, fs, ...
    lle_m, lle_tau, lle_twin, lle_fit_len, ...
    apen_m, apen_r, ...
    cd_dim, cd_tau, ...
    nfft, phaseBins)

feats = NaN(23,1);

% If zscore row is NaN or constant -> return NaNs
if any(isnan(x)) || std(x) < 1e-12
    return;
end

feats(1) = lle_rosenstein_full(x, fs, lle_m, lle_tau, lle_twin, lle_fit_len);
feats(2) = hurst_rs(x);
feats(3) = apen_full(x, apen_m, apen_r);
feats(4) = corr_dim_gp_full(x, cd_dim, cd_tau);

idx = 5;
for k = 3:8
    feats(idx) = mean(x.^k);
    idx = idx + 1;
end

hos = hos_features_bispec_bicep_2d(x, nfft, phaseBins);
feats(11:23) = hos(:);
end

%% =========================================================================
% LLE — Rosenstein (FULL)
%% =========================================================================
function lle = lle_rosenstein_full(x, fs, m, tau, t_win, fit_len)
N = numel(x);
M = N - (m-1)*tau;
if M < (t_win + fit_len + 5)
    lle = NaN; return;
end

Y = zeros(M,m);
for k = 1:m
    Y(:,k) = x(1+(k-1)*tau : M+(k-1)*tau);
end

maxI = M - (t_win + fit_len);
if maxI < 10
    lle = NaN; return;
end
seedIdx = 1:maxI;

dlog = NaN(numel(seedIdx), fit_len);
validCnt = 0;

for si = 1:numel(seedIdx)
    i = seedIdx(si);
    p = Y(i,:);

    D = sum((Y - p).^2, 2);

    lo = max(1, i - t_win);
    hi = min(M, i + t_win);
    D(lo:hi) = Inf;

    [~, j] = min(D);
    if isinf(D(j)) || (i+fit_len > M) || (j+fit_len > M)
        continue;
    end

    validCnt = validCnt + 1;
    for k = 1:fit_len
        dist_k = norm(Y(i+k,:) - Y(j+k,:));
        dlog(validCnt,k) = log(max(dist_k, 1e-12));
    end
end

if validCnt < 20
    lle = NaN; return;
end

dlog = dlog(1:validCnt,:);
t = (1:fit_len)/fs;
p = polyfit(t, mean(dlog,1,'omitnan'), 1);
lle = p(1);
end

%% =========================================================================
% HURST (R/S)
%% =========================================================================
function h = hurst_rs(x)
N = numel(x);
if N < 200
    h = NaN; return;
end

wins = [floor(N/4), floor(N/8), floor(N/16)];
wins = wins(wins >= 20);
if numel(wins) < 2
    h = NaN; return;
end

RS = zeros(size(wins));
for k = 1:numel(wins)
    n = wins(k);
    ns = floor(N/n);
    X = reshape(x(1:n*ns), n, ns);
    X = X - mean(X,1);
    Y = cumsum(X,1);
    R = max(Y,[],1) - min(Y,[],1);
    S = std(X,0,1) + eps;
    RS(k) = mean(R./S);
end

p = polyfit(log(wins), log(RS), 1);
h = p(1);
end

%% =========================================================================
% ApEn FULL (exact definition, O(N^2))
%% =========================================================================
function apen = apen_full(x, m, r)
N = numel(x);
if N < 50
    apen = NaN; return;
end

phi = zeros(1,2);
for kk = 0:1
    mm = m + kk;
    L  = N - mm + 1;
    if L <= 5
        apen = NaN; return;
    end

    pat = zeros(L, mm);
    for j = 1:mm
        pat(:,j) = x(j:j+L-1);
    end

    Ci = zeros(L,1);
    for i = 1:L
        v = pat(i,:);
        d = max(abs(pat - v), [], 2);
        Ci(i) = mean(d <= r);
    end

    phi(kk+1) = mean(log(Ci + eps));
end

apen = phi(1) - phi(2);
end

%% =========================================================================
% Correlation Dimension (GP) FULL
%% =========================================================================
function cd = corr_dim_gp_full(x, dim, tau)
N = numel(x);
M = N - (dim-1)*tau;
if M < 50
    cd = NaN; return;
end

Y = zeros(M,dim);
for k = 1:dim
    Y(:,k) = x(1+(k-1)*tau : M+(k-1)*tau);
end

d = pdist(Y);
d = d(d>0);

if numel(d) < 200
    cd = NaN; return;
end

rmin = max(min(d), 1e-12);
rmax = max(d);

nR = 18;
r = logspace(log10(rmin), log10(rmax), nR);

Cr = zeros(size(r));
for i = 1:numel(r)
    Cr(i) = mean(d < r(i));
end

valid = Cr > 0 & Cr < 1;
if sum(valid) < 6
    cd = NaN; return;
end

p = polyfit(log(r(valid)), log(Cr(valid)), 1);
cd = p(1);
end

%% =========================================================================
% HOS
%% =========================================================================
function hos = hos_features_bispec_bicep_2d(x, nfft, phaseBins)
X = fft(x, nfft);
X = X(1:(nfft/2+1));
F = numel(X);

Bmag = zeros(F,F);
Bph  = zeros(F,F);
mask = false(F,F);

for f1 = 1:F
    for f2 = 1:f1
        f3 = f1 + f2 - 1;
        if f3 <= F
            val = X(f1)*X(f2)*conj(X(f3));
            Bmag(f1,f2) = abs(val);
            Bph(f1,f2)  = angle(val);
            mask(f1,f2) = true;
        end
    end
end

Vmag = Bmag(mask);
Vlog = log(Vmag + 1e-12);

diagIdx = false(F,1);
for f = 1:F
    if (2*f - 1) <= F
        diagIdx(f) = true;
    end
end

Vdiag = zeros(sum(diagIdx),1);
cnt = 0;
for f = 1:F
    if diagIdx(f)
        cnt = cnt + 1;
        Vdiag(cnt) = log(Bmag(f,f) + 1e-12);
    end
end

H1 = sum(Vlog);
H2 = sum(Vdiag);

n = (1:numel(Vdiag))';
sd = sum(Vdiag) + eps;
H3 = sum(n .* Vdiag) / sd;
H4 = sum(((n - H3).^2) .* Vdiag) / sd;

[iIdx,jIdx] = find(mask);
w = sqrt(double(iIdx).^2 + double(jIdx).^2);

sp = sum(Vlog) + eps;
H5 = sum(w .* Vlog) / sp;

P = Vmag / (sum(Vmag) + eps);
BE1 = -sum(P .* log(P + eps));
BE2 = -sum((P.^2) .* log(P.^2 + eps));

Vph = Bph(mask);
if isempty(Vph) || all(~isfinite(Vph))
    BEp = NaN;
else
    [counts,~] = histcounts(Vph, phaseBins);
    pp = counts / (sum(counts) + eps);
    pp(pp==0) = [];
    if isempty(pp), BEp = NaN; else, BEp = -sum(pp .* log(pp)); end
end

Lmat = zeros(F,F);
Lmat(mask) = Vlog;

Cb = real(ifft2(Lmat));
CbMag = abs(Cb);

CbV = CbMag(mask);

CbDiag = zeros(sum(diagIdx),1);
cnt = 0;
for f = 1:F
    if diagIdx(f)
        cnt = cnt + 1;
        CbDiag(cnt) = CbMag(f,f);
    end
end

BH1 = sum(CbV);
BH2 = sum(CbDiag);

nd = (1:numel(CbDiag))';
sd2 = sum(CbDiag) + eps;
BH3 = sum(nd .* CbDiag) / sd2;
BH4 = sum(((nd - BH3).^2) .* CbDiag) / sd2;

sp2 = sum(CbV) + eps;
BH5 = sum(w .* CbV) / sp2;

hos = [H1;H2;H3;H4;H5; BH1;BH2;BH3;BH4;BH5; BE1;BE2;BEp];
end

%% =========================================================================
% IO
%% =========================================================================
function write_csv_header(path, headers)
fid = fopen(path,'w');
if fid<0, error('Cannot open output file for writing.'); end
for i = 1:numel(headers)
    if i < numel(headers)
        fprintf(fid, '%s,', headers{i});
    else
        fprintf(fid, '%s\n', headers{i});
    end
end
fclose(fid);
end

function append_numeric_rows_csv(path, M)
fid = fopen(path,'a');
if fid<0, error('Cannot open output file for appending.'); end
[nr,nc] = size(M);
fmt = [repmat('%.10g,',1,nc-1), '%.10g\n'];
for r = 1:nr
    fprintf(fid, fmt, M(r,:));
end
fclose(fid);
end

%% =========================================================================
%% =========================================================================
function meta = parse_filename_paper(fname)
meta.SubjectID = NaN;
meta.Group     = NaN;
meta.BandType  = 0;

tok = regexp(fname, '^(\d+)_', 'tokens', 'once');
if isempty(tok), return; end
meta.SubjectID = str2double(tok{1});

if contains(fname,'_PD_','IgnoreCase',true)
    meta.Group = 1;
elseif contains(fname,'_HC_','IgnoreCase',true) || contains(fname,'_CTL_','IgnoreCase',true)
    meta.Group = 0;
else
    meta.Group = NaN;
end

if contains(fname,'AllBands','IgnoreCase',true)
    meta.BandType = 1;
elseif contains(fname,'Alpha','IgnoreCase',true)
    meta.BandType = 2;
elseif contains(fname,'Beta','IgnoreCase',true)
    meta.BandType = 3;
else
    meta.BandType = 0;
end
end

function s = sanitize_label(lbl)
s = regexprep(lbl, '\s+', '');
s = regexprep(s, '[^a-zA-Z0-9_]', '_');
end