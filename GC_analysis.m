clear;
close;
clc;
% Addpath and check compilation of mex file.
run('~/GC_clean/startup.m'); 

% load data.
path = 'data/';
load(strcat(path,'preprocessed_data.mat'));

data_series_cell = {data_series_raw', data_series_delta',...
                    data_series_theta', data_series_alpha',...
                    data_series_beta', data_series_gamma',...
                    data_series_high_gamma', data_series_sub_delta',...
                    data_series_above_delta'};

band_cell = {'raw', 'delta', 'theta', 'alpha',...
             'beta', 'gamma', 'high_gamma', ...
             'sub_delta', 'above_delta'};

for i=1:9
  try
    % Choose a fitting order for GC.
    X = data_series_cell{i};
    len = length(X);
    % od_max = 100;
    % [od_joint, od_vec] = chooseOrderFull(X, 'AICc', od_max);
    % m = max([od_joint, od_vec])
    % For a fast schematic test, use:
    m = chooseOrderAuto(X, 'BIC')

    % The Granger Causality value (in matrix form).
    % GC = nGrangerTfast(X, m);
    GC = pos_nGrangerT_qrm(X, m);

    % Significance test: Non-zero probably based on 0-hypothesis (GC==0).
    p_nonzero = gc_prob_nonzero(GC, m, len);

    % The connectivity matrix.
    p_value = 0.0001;
    % net_adjacency = p_nonzero > 1 - p_value;

    % This should give the same result.
    gc_zero_line = chi2inv(1-p_value, m)/len;
    % net_adjacency2 = GC > gc_zero_line;

    % save data
    save(strcat(path,'m_data_', band_cell{i}, '.mat'), 'm', 'GC', 'p_nonzero', 'gc_zero_line');
  catch ME
    ME.message
    disp(strcat(band_cell{i}, ' failed to extimate GC'))
    continue
  end
end