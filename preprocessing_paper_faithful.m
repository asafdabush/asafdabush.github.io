%% Preprocessing - Paper Faithful (AAR + FastICA) | Save IC datasets (All/Alpha/Beta)
% This script replicates the preprocessing pipeline described in the paper
% "Diagnosis of Parkinson's disease using higher order statistical analysis of alpha and beta rhythms".
% It is faithful to the methodology described in the paper and uses EEGLAB functions
% to perform artifact removal (AAR), independent component analysis (ICA), resampling,
% filtering, and segmentation. The script operates on .mat files containing EEG
% structures and saves segmented ICA datasets for subsequent feature extraction.

clc; clear; close all;

base_folder   = 'C:\Users\asafd\Downloads\OneDrive_2025-12-10\PD REST';
output_folder = fullfile(base_folder, 'Preprocessed_Article_ICA_PaperFaithful');
if ~exist(output_folder,'dir'), mkdir(output_folder); end

[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab; %#ok<ASGLU>

% Paper parameters
params.target_srate = 500;
params.seg_len_sec  = 2;
params.seg_samples  = params.target_srate * params.seg_len_sec; % 1000 samples per segment
params.filter_all   = [0.1, 100];
params.filter_alpha = [8, 12];
params.filter_beta  = [13, 30];

file_list = dir(fullfile(base_folder,'*.mat'));
if isempty(file_list), error('No .mat files found.'); end

fprintf('Found %d files. Starting...\n', numel(file_list));

for i = 1:numel(file_list)
    filename = file_list(i).name;
    filepath = fullfile(base_folder, filename);
    [~, subject_name, ~] = fileparts(filename);

    fprintf('\n-----------------------------------------------\n');
    fprintf('Processing %d/%d: %s\n', i, numel(file_list), subject_name);

    %% 1) Load EEG dataset
    try
        loaded = load(filepath,'EEG');
        if isfield(loaded,'EEG')
            EEG = loaded.EEG;
        else
            tmp = load(filepath);
            fn = fieldnames(tmp);
            EEG = tmp.(fn{1});
        end
    catch ME
        warning('Load failed: %s', ME.message);
        continue;
    end
    if ~isfield(EEG,'data') || isempty(EEG.data)
        warning('Empty EEG.data. Skipping.');
        continue;
    end

    EEG.setname = subject_name;

    %% 2) Remove head position channels (X/Y/Z)
    try
        EEG = pop_select(EEG,'nochannel',{'X','Y','Z'});
    catch
        % Fallback: if dataset contains extra channels at positions 65-67
        if size(EEG.data,1) >= 67
            EEG = pop_select(EEG,'nochannel',[65 66 67]);
        end
    end

    %% 3) Fix metadata and resample to 500 Hz if necessary
    if ~isfield(EEG,'srate') || isempty(EEG.srate), EEG.srate = params.target_srate; end
    EEG.nbchan = size(EEG.data,1);
    EEG.pnts   = size(EEG.data,2);
    EEG.trials = 1;
    EEG.xmin   = 0;
    EEG.xmax   = (EEG.pnts-1)/EEG.srate;

    if EEG.srate ~= params.target_srate
        fprintf('   -> Resampling to %d Hz...\n', params.target_srate);
        EEG = pop_resample(EEG, params.target_srate);
    end
    EEG = eeg_checkset(EEG);

    %% 4) Load channel locations (helps AAR/ICA)
    loc_file = 'standard-10-5-cap385.elp';
    paths = {which(loc_file), fullfile(fileparts(which('eeglab.m')), 'plugins','dipfit','standard_BEM',loc_file)};
    found_loc = '';
    for k=1:numel(paths)
        if ~isempty(paths{k}) && exist(paths{k},'file')
            found_loc = paths{k}; break;
        end
    end
    if ~isempty(found_loc)
        try
            EEG = pop_chanedit(EEG,'lookup',found_loc);
        catch
            warning('Failed loading channel locations.');
        end
    end

    %% 5) IIR bandpass 0.1–100 Hz (Butterworth + filtfilt)
    fprintf('   -> IIR bandpass 0.1–100 Hz...\n');
    EEG.data = iir_bandpass(EEG.data, EEG.srate, params.filter_all);

    %% 6) AAR: EOG and EMG artifact removal
    if exist('pop_autobsseog','file')
        fprintf('   -> AAR EOG removal...\n');
        try
            EEG = pop_autobsseog(EEG, [], [], 'sobi', {'eigratio',[1e6]}, 'eog_corr', {'range',[2 20]});
        catch ME
            warning('EOG removal failed: %s', ME.message);
        end
    else
        warning('AAR EOG function missing.');
    end

    if exist('pop_autobssemg','file')
        fprintf('   -> AAR EMG removal...\n');
        try
            EEG = pop_autobssemg(EEG, [], [], 'sobi', {'eigratio',[1e6]}, 'emg_psd', {'ratio',[10],'range',[15 30]});
        catch ME
            warning('EMG removal failed: %s', ME.message);
        end
    else
        warning('AAR EMG function missing.');
    end
    EEG = eeg_checkset(EEG);

    %% 7) FastICA decomposition
    fprintf('   -> FastICA...\n');
    try
        EEG = pop_runica(EEG, 'icatype','fastica', 'approach','symm', 'g','tanh');
    catch
        try
            EEG = pop_runica(EEG, 'icatype','runica');
        catch ME
            warning('ICA failed: %s', ME.message);
            continue;
        end
    end

    %% 8) Build ICA activations explicitly and replace EEG.data with IC signals
    W = double(EEG.icaweights) * double(EEG.icasphere);  % nIC x nChan
    X = double(EEG.data);                                % nChan x nSamp
    IC = W * X;                                          % nIC x nSamp

    EEG_IC = EEG;
    EEG_IC.data   = IC;
    EEG_IC.nbchan = size(IC,1);
    EEG_IC.pnts   = size(IC,2);
    EEG_IC.trials = 1;

    % Replace channel labels to IC1..ICn to avoid confusion
    EEG_IC.chanlocs = struct('labels', cell(1,EEG_IC.nbchan));
    for c = 1:EEG_IC.nbchan
        EEG_IC.chanlocs(c).labels = sprintf('IC%d', c);
    end
    EEG_IC = eeg_checkset(EEG_IC);

    %% 9) Save segmented datasets (All/Alpha/Beta) from IC signals
    % All frequency bands
    save_segments_exact1000(EEG_IC, params.seg_samples, output_folder, [subject_name '_AllBands']);

    % Alpha band (8–12 Hz)
    EEG_Alpha = EEG_IC;
    EEG_Alpha.data = iir_bandpass(EEG_IC.data, params.target_srate, params.filter_alpha);
    save_segments_exact1000(EEG_Alpha, params.seg_samples, output_folder, [subject_name '_Alpha']);

    % Beta band (13–30 Hz)
    EEG_Beta = EEG_IC;
    EEG_Beta.data = iir_bandpass(EEG_IC.data, params.target_srate, params.filter_beta);
    save_segments_exact1000(EEG_Beta, params.seg_samples, output_folder, [subject_name '_Beta']);

    EEG = [];
end

fprintf('\n*** DONE (Paper-faithful ICA datasets saved)! ***\n');

%% ---------------- Helper functions ----------------
function d_out = iir_bandpass(d_in, srate, band)
    % Butterworth bandpass filter with filtfilt for zero-phase distortion.
    [b, a] = butter(4, band / (srate/2), 'bandpass');
    tmp = double(d_in);
    d_out = zeros(size(tmp), 'like', tmp);
    for c = 1:size(tmp,1)
        d_out(c,:) = filtfilt(b, a, tmp(c,:));
    end
    if isa(d_in,'single'), d_out = single(d_out); end
end

function save_segments_exact1000(EEG, samps, out_path, out_name)
    % Segment EEG into epochs of exactly 1000 samples and save as .set files.
    if EEG.pnts < samps, return; end
    n_segs = floor(EEG.pnts / samps);
    if n_segs < 1, return; end

    EEG.event = [];
    lats = 1:samps:(n_segs*samps);
    for k = 1:numel(lats)
        EEG.event(k).type    = 'S';
        EEG.event(k).latency = lats(k);
        EEG.event(k).urevent = k;
    end

    try
        EEG_Ep = pop_epoch(EEG, {'S'}, [0 samps/EEG.srate], 'epochinfo', 'yes');
        EEG_Ep.data = EEG_Ep.data(:,1:samps,:);
        EEG_Ep.pnts = samps;
        EEG_Ep.xmax = (samps-1)/EEG_Ep.srate;

        EEG_Ep = pop_rmbase(EEG_Ep, []);
        pop_saveset(EEG_Ep, 'filename', [out_name '.set'], 'filepath', out_path);
    catch ME
        warning('Save failed for %s: %s', out_name, ME.message);
    end
end