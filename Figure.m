
% Note(2024/2): ----------------

% This script is designed to visualize datasets related to specific figures and experiments described in our manuscript.
% It aims to highlight data points depicted in various figures, each associated with its respective figure number.
% The script selectively omits data points whose positions or values are directly presented within the manuscript's figures (e.g., Fig 2E, S3A, S6, S9C, S10A & S10B, S14A),
% thereby focusing on the visualization of data not compeletely elaborated in the figures or the main body of the text.
% Firing rate data, somewhere represented as responses, is calculated with a 500 Hz sampling rate, translating to spike counts every 2 milliseconds.
% Code formatting has been optimized for backward compatibility with MATLAB versions as early as MATLAB 2017a.
% The structure of this script supports standalone execution for specific figure visualizations, contingent upon having the requisite data workspace preloaded.
% It relies solely on MATLAB's built-in functions and does not require any external functions to be added to the MATLAB path.

% ---------------------

%% Fig.1C Time course of neuronal activity from an example electrode site (DP)

% Extract relevant data from the input matrix
time = Exp_For_Mem_Encoding(:, 1); % Time points
Cue90_mean = Exp_For_Mem_Encoding(:, 2); % Mean values for Cue90 condition
Cue90_sem = Exp_For_Mem_Encoding(:, 3); % Standard errors for Cue90 condition
Cue180_mean = Exp_For_Mem_Encoding(:, 4); % Mean values for Cue180 condition
Cue180_sem = Exp_For_Mem_Encoding(:, 5); % Standard errors for Cue180 condition
t_test_results = Exp_For_Mem_Encoding(:, 6); % t-test results

figure;

subplot(1,3,1:2)
hold on;

% Plot the mean values for Cue90 and Cue180 conditions over time
plot(time, Cue90_mean, 'r', 'LineWidth', 2);
plot(time, Cue180_mean, 'b', 'LineWidth', 2);

% Plot the standard error of the mean (SEM) as shaded areas around the mean curves
fill([time', fliplr(time')], [Cue90_mean' + Cue90_sem', fliplr(Cue90_mean' - Cue90_sem')], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
fill([time', fliplr(time')], [Cue180_mean' + Cue180_sem', fliplr(Cue180_mean' - Cue180_sem')], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Plot black squares to indicate significant t-test results
sig_threshold = 0.2; % Set significance threshold
VisualizeLevel = 0.25; % Set height for displaying significant markers
significantPoints = find(t_test_results >= sig_threshold); % Find significant time points
plot(time(significantPoints), repmat(VisualizeLevel, length(significantPoints), 1), 'ks', 'MarkerSize', 4);

xlim([-150, 1700]); % Set x-axis range
xlabel('Time (ms)'); % Set x-axis label
ylabel('Mean Response'); % Set y-axis label
title('Mean Response with SEM for Cued 90/180'); % Set plot title
legend('Cued 90', 'Cued 180', 'Location', 'Best'); % Set legend
hold off; % Finish plotting multiple elements in the same figure

% Behavior compare
Cue90_behavior = Behavior_For_90_180(:, 1); % Cue90 condition data
Cue180_behavior = Behavior_For_90_180(:, 2); % Cue180 condition data

subplot(1,3,3)

% Combine the data from both conditions for boxplot
combined_data = [Cue90_behavior, Cue180_behavior];

% Create a boxplot to compare the two conditions
boxplot(combined_data, 'Labels', {'Cued 90', 'Cued 180'});

ylabel('Behavioral Data'); % Set y-axis label
title('Comparison of Cued 90 and Cued 180 Conditions'); % Set plot title

%% Fig.1D Neuronal activity from the same electrode for incorrect trials & fixation task (lower demanding mnemonic behavior) trials

% Extract data
time = Exp_For_MemInc_Encoding(:, 1);
MemInc_Cue90_mean = Exp_For_MemInc_Encoding(:, 2);
MemInc_Cue90_sem = Exp_For_MemInc_Encoding(:, 3);
MemInc_Cue180_mean = Exp_For_MemInc_Encoding(:, 4);
MemInc_Cue180_sem = Exp_For_MemInc_Encoding(:, 5);
MemInc_t_test_results = Exp_For_MemInc_Encoding(:, 6);

LessDemand_Cue90_mean = Exp_For_Lessdemand_Encoding(:, 2);
LessDemand_Cue90_sem = Exp_For_Lessdemand_Encoding(:, 3);
LessDemand_Cue180_mean = Exp_For_Lessdemand_Encoding(:, 4);
LessDemand_Cue180_sem = Exp_For_Lessdemand_Encoding(:, 5);
LessDemand_t_test_results = Exp_For_Lessdemand_Encoding(:, 6);

figure;

% Create subplots for the two datasets
for i = 1:2
    % Select the data for the current subplot
    if i == 1
        Cue90_mean = MemInc_Cue90_mean;
        Cue90_sem = MemInc_Cue90_sem;
        Cue180_mean = MemInc_Cue180_mean;
        Cue180_sem = MemInc_Cue180_sem;
        t_test_results = MemInc_t_test_results;
        subplot_title = 'Incorrect trials';
    else
        Cue90_mean = LessDemand_Cue90_mean;
        Cue90_sem = LessDemand_Cue90_sem;
        Cue180_mean = LessDemand_Cue180_mean;
        Cue180_sem = LessDemand_Cue180_sem;
        t_test_results = LessDemand_t_test_results;
        subplot_title = 'lower Demanding trials';
    end
    subplot(1, 2, i);
    hold on;
    
    % Plot the mean values for Cue90 and Cue180 conditions over time
    plot(time, Cue90_mean, 'r', 'LineWidth', 2);
    plot(time, Cue180_mean, 'b', 'LineWidth', 2);
    
    % Plot the standard error of the mean (SEM) as shaded areas around the mean curves
    % for the Cue90 and Cue180 conditions
    fill([time', fliplr(time')], [Cue90_mean' + Cue90_sem', fliplr(Cue90_mean' - Cue90_sem')], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    fill([time', fliplr(time')], [Cue180_mean' + Cue180_sem', fliplr(Cue180_mean' - Cue180_sem')], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    
    % Plot black squares to indicate significant t-test results
    sig_threshold = 0.2; % Set significance threshold
    VisualizeLevel = 0.25; % Set height for displaying significant markers
    significantPoints = find(t_test_results >= sig_threshold); % Find significant time points
    plot(time(significantPoints), repmat(VisualizeLevel, length(significantPoints), 1), 'ks', 'MarkerSize', 4);
    
    xlim([-150, 1700]); % Set x-axis range
    xlabel('Time (ms)'); % Set x-axis label
    ylabel('MUA (Sp/s)'); % Set y-axis label
    title(subplot_title); % Set plot title
    legend('Cue90', 'Cue180', 'Location', 'Best'); % Set legend
    hold off; % Finish plotting
end

%% Fig.1E Data for three conditions across all valid electrodes in monkey DP

% Extract x (highest) and y (lowest) values from the data matrices
Memcorr_x = Memcorr_HL(:, 1);
Memcorr_y = Memcorr_HL(:, 2);
MemInc_x = MemInc_HL(:, 1);
MemInc_y = MemInc_HL(:, 2);
LessDem_x = LessDem_HL(:, 1);
LessDem_y = LessDem_HL(:, 2);

% Calculate CMI for each condition
CMI_Memcorr = (Memcorr_x - Memcorr_y) ./ (Memcorr_y + Memcorr_x);
CMI_MemInc = (MemInc_x - MemInc_y) ./ (MemInc_y + MemInc_x);
CMI_LessDem = (LessDem_x - LessDem_y) ./ (LessDem_y + LessDem_x);

figure;

subplot(1, 3, 1);
hold on;

% Plot MemInc_HL data using gray filled squares
scatter(MemInc_x, MemInc_y, 'filled', 'MarkerFaceColor', [0.5, 0.5, 0.5]);

% Plot LessDem_HL data using blue filled diamonds
scatter(LessDem_x, LessDem_y, 'b', 'filled', 'd');

% Add a dashed diagonal line
diag_min = min([Memcorr_x; MemInc_x; LessDem_x]);
diag_max = max([Memcorr_y; MemInc_y; LessDem_y]);
plot([diag_min, diag_max], [diag_min, diag_max], 'k--');

% Plot Memcorr_HL data using red filled circles (in front)
scatter(Memcorr_x, Memcorr_y, 'r', 'filled');

xlabel('Lowest Response');
ylabel('Highest Response');
legend('MemInc', 'Fixation', 'Diagonal Line', 'Memcorr', 'Location', 'best');
hold off;

% CMI histograms
subplot(1, 3, 2);
hold on;

% Calculate histograms
[n_CMI_Memcorr, edges_CMI_Memcorr] = histcounts(CMI_Memcorr, 'BinMethod', 'fd');
[n_CMI_MemInc, edges_CMI_MemInc] = histcounts(CMI_MemInc, 'BinMethod', 'fd');
[n_CMI_LessDem, edges_CMI_LessDem] = histcounts(CMI_LessDem, 'BinMethod', 'fd');

% Convert histogram counts to relative values for better visibility
n_CMI_Memcorr = n_CMI_Memcorr / sum(n_CMI_Memcorr);
n_CMI_MemInc = n_CMI_MemInc / sum(n_CMI_MemInc);
n_CMI_LessDem = n_CMI_LessDem / sum(n_CMI_LessDem);

% Calculate bin centers and mean values
bin_centers_CMI_Memcorr = (edges_CMI_Memcorr(1:end-1) + edges_CMI_Memcorr(2:end)) / 2;
bin_centers_CMI_MemInc = (edges_CMI_MemInc(1:end-1) + edges_CMI_MemInc(2:end)) / 2;
bin_centers_CMI_LessDem = (edges_CMI_LessDem(1:end-1) + edges_CMI_LessDem(2:end)) / 2;
mean_CMI_Memcorr = mean(CMI_Memcorr);
mean_CMI_MemInc = mean(CMI_MemInc);
mean_CMI_LessDem = mean(CMI_LessDem);

% Plot histograms
stairs(bin_centers_CMI_MemInc, n_CMI_MemInc, 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2); % Plot CMI_MemInc histogram
stairs(bin_centers_CMI_LessDem, n_CMI_LessDem, 'b', 'LineWidth', 2); % Plot CMI_LessDem histogram
stairs(bin_centers_CMI_Memcorr, n_CMI_Memcorr, 'r', 'LineWidth', 2); % Plot CMI_Memcorr histogram (in front)

% Add mean values as vertical lines
plot([mean_CMI_MemInc, mean_CMI_MemInc], [0, max([n_CMI_Memcorr, n_CMI_MemInc, n_CMI_LessDem])], 'Color', [0.5, 0.5, 0.5], 'LineStyle', '--');
plot([mean_CMI_LessDem, mean_CMI_LessDem], [0, max([n_CMI_Memcorr, n_CMI_MemInc, n_CMI_LessDem])], 'b--');
plot([mean_CMI_Memcorr, mean_CMI_Memcorr], [0, max([n_CMI_Memcorr, n_CMI_MemInc, n_CMI_LessDem])], 'r--');

xlabel('CMI');
ylabel('Relative count');
title('Distribution of CMI for all Conditions');

legend('MemInc', 'LessDem', 'Memcorr', 'Location', 'best');

hold off;

subplot(1, 3, 3);
hold on;

% Plot MemInc_HL_normalized data using gray filled squares
scatter(MemInc_x_normalized, MemInc_y_normalized, 'filled', 'MarkerFaceColor', [0.5, 0.5, 0.5]);

% Plot LessDem_HL_normalized data using blue filled diamonds
scatter(LessDem_x_normalized, LessDem_y_normalized, 'b', 'filled', 'd');

% Add a dashed diagonal line
diag_min = min([Memcorr_x_normalized; MemInc_x_normalized; LessDem_x_normalized]);
diag_max = max([Memcorr_y_normalized; MemInc_y_normalized; LessDem_y_normalized]);
plot([diag_min, diag_max], [diag_min, diag_max], 'k--');

% Plot Memcorr_HL_normalized data using red filled circles (in front)
scatter(Memcorr_x_normalized, Memcorr_y_normalized, 'r', 'filled');

xlabel('Lowest Response');
ylabel('Highest Response');
legend('MemInc_normalized', 'Fixation', 'Diagonal Line', 'Memcorr_normalized', 'Location', 'best');
hold off;

%% Fig.1F CMIs for both monkeys in three conditions, combined across all valid electrode sites

DP_CMI_Memory = CMI_DP(:,1);
DP_CMI_Memory_Inc = CMI_DP(:,2);
DP_CMI_Memory_Fixing = CMI_DP(:,3);

DQ_CMI_Memory = CMI_DQ(:,1);
DQ_CMI_Memory_Inc = CMI_DQ(:,2);
DQ_CMI_Memory_Fixing = CMI_DQ(:,3);

Avg_DP_baseline = mean(CMI_BaselDP);
Avg_DQ_baseline = mean(CMI_BaselDQ);

% Perform t-tests
[~, p_DP_1_2] = ttest(DP_CMI_Memory, DP_CMI_Memory_Inc);
[~, p_DP_2_3] = ttest(DP_CMI_Memory_Inc, DP_CMI_Memory_Fixing);
[~, p_DP_1_3] = ttest(DP_CMI_Memory, DP_CMI_Memory_Fixing);

[~, p_DQ_1_2] = ttest(DQ_CMI_Memory, DQ_CMI_Memory_Inc);
[~, p_DQ_2_3] = ttest(DQ_CMI_Memory_Inc, DQ_CMI_Memory_Fixing);
[~, p_DQ_1_3] = ttest(DQ_CMI_Memory, DQ_CMI_Memory_Fixing);

% Prepare data for bar plot
DP_data = [mean(DP_CMI_Memory), mean(DP_CMI_Memory_Inc), mean(DP_CMI_Memory_Fixing)];
DQ_data = [mean(DQ_CMI_Memory), mean(DQ_CMI_Memory_Inc), mean(DQ_CMI_Memory_Fixing)];

figure;

x_labels = {'DP', 'DQ'};

% Plot the bar chart
bar_data = [DP_data; DQ_data];
bar_handle = bar(bar_data);
set(gca, 'XTickLabel', x_labels);

% Add baseline average lines
hold on;
baseline_DP = plot(xlim, [Avg_DP_baseline, Avg_DP_baseline], '--', 'Color', 'r', 'LineWidth', 2);
baseline_DQ = plot(xlim, [Avg_DQ_baseline, Avg_DQ_baseline], '--', 'Color', 'b', 'LineWidth', 2);

legend([bar_handle(1), bar_handle(2), bar_handle(3), baseline_DP, baseline_DQ], 'Mem-corr', 'Mem-inc', 'lower-demand', 'DP Baseline', 'DQ Baseline');

xlabel('Animal');
ylabel('CMI value');
title('DP and DQ Data Comparison with Baselines');
% Add significance labels
hold on;

sig_threshold = 0.01;

% Calculate label positions based on the data
max_val = max([DP_data(:); DQ_data(:)]);
min_val = min([DP_data(:); DQ_data(:)]);
label_offset = (max_val - min_val) * 0.05;  % 5% of the data range

% Set label positions for each comparison
y_pos = max_val + label_offset;

if p_DP_1_2 < sig_threshold
    text(0.7, y_pos, 'p < 0.01', 'FontSize', 8);
end
if p_DP_2_3 < sig_threshold
    text(1.7, y_pos, 'p < 0.01', 'FontSize', 8);
end
if p_DP_1_3 < sig_threshold
    text(1, y_pos, 'p < 0.01', 'FontSize', 8);
end

if p_DQ_1_2 < sig_threshold
    text(1.3, y_pos, 'p < 0.01', 'FontSize', 8);
end
if p_DQ_2_3 < sig_threshold
    text(2.3, y_pos, 'p < 0.01', 'FontSize', 8);
end
if p_DQ_1_3 < sig_threshold
    text(1.6, y_pos, 'p < 0.01', 'FontSize', 8);
end

%% Fig.2A & S8 Cross-temporal decoding of VWM contents using separate classifiers

Cue_Acquire = 4;
matrix_list = {DP_Ori_GnrMatrix, DP_Clr_GnrMatrix, DP_Face_GnrMatrix, DQ_Ori_GnrMatrix, DQ_Clr_GnrMatrix, DQ_Face_GnrMatrix};

SR = 1/2; % Sampling rate information
BaselineL = 200; % Baseline length
Encoding_window = ((0:2:200)+BaselineL) *SR; % Encoding visual time window
Maint_window = ((700:2:1700)+BaselineL) *SR; % Encoding memory time window
Windowspan = [100] *SR;
step = 50 *SR;
ve = 1/21.*ones(21,1);

figure;

% Loop through the six matrices and create a subplot for each matrix
for i = 1:6
    Curr_Cross_Avg = matrix_list{i}; % Extract the current matrix from the cell array
    xx = 1-BaselineL:step*(1/SR):size(Curr_Cross_Avg,1)*step*(1/SR)-BaselineL; yy = xx;
    
    subplot(2, 3, i); 
    hold on;
    imagesc(xx, yy, Curr_Cross_Avg);
    caxis([1/Cue_Acquire 1.3/Cue_Acquire]);
    axis([1 size(Curr_Cross_Avg,1)*step*(1/SR)-BaselineL 1 size(Curr_Cross_Avg,1)*step*(1/SR)-BaselineL]);
    set(gca, 'FontName', 'Times New Roman');
    ylabel('Training bin (ms)'); xlabel('Testing bin (ms)');
end

%% Fig.2B Averaged cross-sections for stimulus-period (red) and delay period (blue)

figure;

% Initialize the cumulative values of the encoding and memory phase decoder lines
sum_E = zeros(size(xx)); Combine_E = [];
sum_D = zeros(size(xx)); Combine_D = [];

% Loop through the six matrices and create a subplot for each matrix
for i = 1:6
    Enco_st = 1; Enco_end = 250; Delay_st = 700; Delay_end = 1600;
    Enco_range = find(xx >= Enco_st & xx <= Enco_end);
    Delay_range = find(xx >= Delay_st & xx <= Delay_end);
    Curr_Cross = matrix_list{i};
    Enco_matrix = squeeze(mean(Curr_Cross(:, Enco_range, :), 2));
    Delay_matrix = squeeze(mean(Curr_Cross(:, Delay_range, :), 2));
    
    E = squeeze(mean(Enco_matrix, 2));
    D = squeeze(mean(Delay_matrix, 2));
    
    Combine_E(:,i) = E;
    Combine_D(:,i) = D;
    
    % Accumulate the values of the two sets of lines
    sum_E = sum_E + E;
    sum_D = sum_D + D;
    
    % Draw the original lines in light colors
    hold on
    plot(xx, E, 'r', 'linewidth', 0.8, 'Color', [1, 0, 0, 0.3]);
    hold on
    plot(xx, D, 'b', 'linewidth', 0.8, 'Color', [0, 0, 1, 0.3]);
end

% Calculate the average values of the two sets of lines
avg_E = sum_E / 6;
avg_D = sum_D / 6;

% Draw the thick average lines
hold on
plot(xx, avg_E, 'r', 'linewidth', 2);
hold on
plot(xx, avg_D, 'b', 'linewidth', 2);
legend('Stimulus period', 'Delay period');

set(gca, 'FontName', 'Times New Roman');
ylabel('Decoding accuracy');
xlabel('Testing time (ms)');


%% Fig.2D & S9. Neuronal firing rates during stimulus (peak/stimulus period) and delay epochs in three memory tasks for both monkeys

figure;

standardError = @(data) std(data, 0, 1) ./ sqrt(size(data, 1));

% Lists of matrices and their titles
matrices = {DPOriResp, DPClrResp, DPFaceResp, DQOriResp, DQClrResp, DQFaceResp};
titles = {'DPOriResp', 'DPClrResp', 'DPFaceResp', 'DQOriResp', 'DQClrResp', 'DQFaceResp'};

% Loop through each matrix to plot data
for i = 1:6
    matrix = matrices{i};
    subplot(2, 3, i); 
    
    % Compute mean and standard error for Stimulus and Delay
    barData = [mean(matrix(:,1)), mean(matrix(:,2)), mean(matrix(:,3))];
    errData = [standardError(matrix(:,1)), standardError(matrix(:,2)), standardError(matrix(:,3))];
    
    bar(1:3, barData);
    hold on;
    errorbar(1:3, barData, errData, '.');
    
    % Draw baseline using dashed line
    baseline = mean(matrix(:,4));
    line([0.5, 3.5], [baseline, baseline], 'LineStyle', '--');
    
    xlim([0.5, 3.5]);
    xticks([1, 2, 3]);
    xticklabels({'Peak','Stimulus', 'Delay'});
    ylabel('Firing rate (sp/s)');
    title(titles{i});
end

%% Fig.2F Examples of three pairs of neurons, representing neuron pairs
% that are significant only during the stimulus period, only during the delay period, and significant during both periods.

figure;

% Plotting the data for Only_stimpair
subplot(1, 3, 1);
plot(Only_stimpair(:, 1), Only_stimpair(:, 2), 'b', ... % Response during stimulus period
    Only_stimpair(:, 1), Only_stimpair(:, 3), 'r');    % Response during delay period
title('Only Stimulus Pair');
xlabel('Time Lag (ms)');
ylabel('Normalized Efficacy');
legend('Stimulus', 'Delay');

% Plotting the data for Only_delaypair
subplot(1, 3, 2);
plot(Only_delaypair(:, 1), Only_delaypair(:, 2), 'b', ... % Response during stimulus period
    Only_delaypair(:, 1), Only_delaypair(:, 3), 'r');    % Response during delay period
title('Only Delay Pair');
xlabel('Time Lag (ms)');
ylabel('Normalized Efficacy');
legend('Stimulus', 'Delay');

% Plotting the data for Both_periodpair
subplot(1, 3, 3);
plot(Both_periodpair(:, 1), Both_periodpair(:, 2), 'b', ... % Response during stimulus period
    Both_periodpair(:, 1), Both_periodpair(:, 3), 'r');    % Response during delay period
title('Both Period Pair');
xlabel('Time Lag (ms)');
ylabel('Normalized Efficacy');
legend('Stimulus', 'Delay');

%% Fig.2G Matrix of pairs significant in different periods

figure;
set(gcf, 'Color', 'white');

% Record neuron pairs significant only in the stim period
[rows, cols] = find(stimData == 1 & delayData == 0);
stimPairs = [stimPairs; [rows, cols]];

% Record neuron pairs significant only in the delay period
[rows, cols] = find(stimData == 0 & delayData == 1);
delayPairs = [delayPairs; [rows, cols]];

% Record neuron pairs significant in both periods
[rows, cols] = find(stimData == 1 & delayData == 1);
bothPairs = [bothPairs; [rows, cols]];

comparisonMatrix = zeros(size(stimData));

% Mark four conditions
% Significant only in the stim period (represented by 1)
comparisonMatrix(stimData == 1 & delayData == 0) = 1;
% Significant only in the delay period (represented by 2)
comparisonMatrix(stimData == 0 & delayData == 1) = 2;
% Significant in both periods (represented by 3)
comparisonMatrix(stimData == 1 & delayData == 1) = 3;
% Not significant in either period (represented by 4)
comparisonMatrix(stimData == 0 & delayData == 0) = 4;

% Plotting the comparison matrix
imagesc(comparisonMatrix);
colormap([1 0 0; 0 0 1; 0 1 0; 1 1 1]); % Red, blue, green, and white represent the four conditions
colorbar; % Optional, displays a color bar

%% Fig.2H Proportional comparison of pairs significant in different periods

% Calculate the mean and standard error of the mean (SEM) for each condition
means_DP = mean(DP_FC_por);
sem_DP = std(DP_FC_por) / sqrt(size(DP_FC_por, 1));

means_DQ = mean(DQ_FC_por);
sem_DQ = std(DQ_FC_por) / sqrt(size(DQ_FC_por, 1));

figure;

% Plotting the data for DP_FC_por
subplot(1, 2, 1);
errorbar(1:3, means_DP, sem_DP, 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b');
title('DP FC Proportion');
xlabel('Condition');
ylabel('Proportion');
set(gca, 'XTick', 1:3, 'XTickLabel', {'Only Stimulus', 'Only Delay', 'Both'});
xlim([0 4]);

% Plotting the data for DQ_FC_por
subplot(1, 2, 2);
errorbar(1:3, means_DQ, sem_DQ, 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r');
title('DQ FC Proportion');
xlabel('Condition');
ylabel('Proportion');
set(gca, 'XTick', 1:3, 'XTickLabel', {'Only Stimulus', 'Only Delay', 'Both'});
xlim([0 4]);

%% Fig.3C Behavioral performance across training sessions

% Extract session and accuracy data
DP_sessions = DP_AssoTask_Behavior(:, 1);
DP_accuracy = DP_AssoTask_Behavior(:, 2);
DQ_sessions = DQ_AssoTask_Behavior(:, 1);
DQ_accuracy = DQ_AssoTask_Behavior(:, 2);

figure;

% First subplot: DP_AssoTask_Behavior
subplot(1, 2, 1);
hold on;
plot(DP_sessions, DP_accuracy, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 6);
set(gca, 'FontName', 'Times New Roman');
ylabel('Accuracy');
xlabel('Session');
title('DP AssoTask Behavior');
hold off;

% Second subplot: DQ_AssoTask_Behavior
subplot(1, 2, 2);
hold on;
plot(DQ_sessions, DQ_accuracy, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6);
set(gca, 'FontName', 'Times New Roman');
ylabel('Accuracy');
xlabel('Session');
title('DQ AssoTask Behavior');
hold off;

%% Fig.3D&F Trial-averaged neuronal activity time course from an example electrode site...
% during early, late training stages & color-color task (CCT) &
% orientation-orientation task (OOT)

% Extract time and Firing Rate from the data matrices
time = Exp_Early_Asso(:, 1);
early_cond1 = Exp_Early_Asso(:, 2);
early_cond2 = Exp_Early_Asso(:, 3);
late_cond1 = Exp_Late_Asso(:, 2);
late_cond2 = Exp_Late_Asso(:, 3);
color_cond1 = Exp_Color_color(:, 2);
color_cond2 = Exp_Color_color(:, 3);
ori_cond1 = Exp_Ori_Ori(:, 2);
ori_cond2 = Exp_Ori_Ori(:, 3);

% Calculate the minimum and maximum values for all curves
min_value = min([early_cond1; early_cond2; late_cond1; late_cond2; color_cond1; color_cond2; ori_cond1; ori_cond2]) * 0.9;
max_value = max([early_cond1; early_cond2; late_cond1; late_cond2; color_cond1; color_cond2; ori_cond1; ori_cond2]) * 1.1;

figure;

% First subplot: early_cond1 and early_cond2
subplot(1, 4, 1);
hold on
plot(time, early_cond1, 'b', 'linewidth', 2);
plot(time, early_cond2, 'r', 'linewidth', 2);
xlabel('Time (ms)');
ylabel('Firing Rate (sp/s)');
title('Early Association');
legend('90-Blue', '180-Red');
xlim([-150, 1700]); % Set x-axis range
ylim([min_value, max_value]); % Set y-axis range
hold off

% Second subplot: late_cond1 and late_cond2
subplot(1, 4, 2);
hold on
plot(time, late_cond1, 'b', 'linewidth', 2);
plot(time, late_cond2, 'r', 'linewidth', 2);
xlabel('Time (ms)');
ylabel('Firing Rate (sp/s)');
title('Late Association');
legend('90-Blue', '180-Red');
xlim([-150, 1700]); % Set x-axis range
ylim([min_value, max_value]); % Set y-axis range
hold off

% Third subplot: color_cond1 and color_cond2
subplot(1, 4, 3);
hold on
time = Exp_Color_color(:, 1);
plot(time, color_cond1, 'b', 'linewidth', 2);
plot(time, color_cond2, 'r', 'linewidth', 2);
xlabel('Time (ms)');
ylabel('Firing Rate (sp/s)');
title('Color-Color');
legend('Blue', 'Red');
xlim([-150, 1700]); % Set x-axis range
ylim([min_value, max_value]); % Set y-axis range
hold off

% Fourth subplot: ori_cond1 and ori_cond2
subplot(1, 4, 4);
hold on
time = Exp_Ori_Ori(:, 1);
plot(time, ori_cond1, 'b', 'linewidth', 2);
plot(time, ori_cond2, 'r', 'linewidth', 2);
xlabel('Time (ms)');
ylabel('Firing Rate (sp/s)');
title('Orientation-Orientation');
legend('Orientation 1', 'Orientation 2');
xlim([-150, 1700]); % Set x-axis range
ylim([min_value, max_value]); % Set y-axis range
hold off

%% Fig.3E VWM content representation (CMI) shift during association task training

% Compute the means and standard errors
mean_DP = mean(DP_Asso_CMI_shift, 1);
sem_DP = std(DP_Asso_CMI_shift, [], 1) / sqrt(size(DP_Asso_CMI_shift, 1));
mean_DQ = mean(DQ_Asso_CMI_shift, 1);
sem_DQ = std(DQ_Asso_CMI_shift, [], 1) / sqrt(size(DQ_Asso_CMI_shift, 1));

figure
% Plot the data for DP_Asso_CMI_shift
subplot(1,2,1)
bar([mean_DP(1:2); mean_DP(3:4)], 'grouped');
hold on;
numgroups = 2;
numbars = 2;
groupwidth = min(0.8, numbars/(numbars + 1.5));
for i = 1:numbars
    x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);
    errorbar(x, [mean_DP(i), mean_DP(i+2)], [sem_DP(i), sem_DP(i+2)], 'k', 'linestyle', 'none', 'linewidth', 1);
end
hold off;
set(gca, 'XTickLabel', {'Early', 'Late'});
ylabel('Mean Response');
title('DP Asso CMI shift');
legend('Stim', 'Delay', 'Location', 'Best');

% Plot the data for DQ_Asso_CMI_shift
subplot(1,2,2);
bar([mean_DQ(1:2); mean_DQ(3:4)], 'grouped');
hold on;
for i = 1:numbars
    x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);
    errorbar(x, [mean_DQ(i), mean_DQ(i+2)], [sem_DQ(i), sem_DQ(i+2)], 'k', 'linestyle', 'none', 'linewidth', 1);
end
hold off;
set(gca, 'XTickLabel', {'Early', 'Late'});
ylabel('Mean Response');
title('DQ Asso CMI shift');
legend('Stim', 'Delay', 'Location', 'Best');

%% Fig.3G & S13 Projections of the CMIs onto the first three PCs

% Set the starting and ending points for the plot
StartP = 375; EndP = 700;
StartP2 = 700; EndP2 = 950;

% Define the data groups
dataGroups = {
    {'DP_CMI_ori_Green', 'DP_CMI_ori_Old', 'DP_CMI_clr_Cont', 'DP_CMI_OOT_Cont'},
    {'DQ_CMI_ori_Green', 'DQ_CMI_ori_Old', 'DQ_CMI_clr_Cont', 'DQ_CMI_OOT_Cont'},
    {'DPc_CMI_ori_Green', 'DPc_CMI_ori_Old', 'DPc_CMI_clr_Cont', 'DPc_CMI_OOT_Cont'},
    {'DQc_CMI_ori_Green', 'DQc_CMI_ori_Old', 'DQc_CMI_clr_Cont', 'DQc_CMI_OOT_Cont'}
    };

% Define the initial view angles for each subplot
viewAngles = [
    -135, -8;
    -100, -8;
    -167, 12;
    -182, 90
    ];

% Define the titles for each subplot
titles = {
    'DP traj',
    'DQ traj',
    'DP traj on color task PC space',
    'DQ traj on color task PC space'
    };

% Create a 3D plot with labeled axes
figure

colors = {'b', 'r', 'g', 'k'}; % Added a color for the fourth line
labels = {'Early', 'Late', 'Clr', 'OOT'}; % Added a label for the fourth line

for i = 1:numel(dataGroups)
    subplot(2, 2, i);
    hold on
    ylabel('PC2'); xlabel('PC1'); zlabel('PC3');
    view(viewAngles(i, :)); 
    title(titles{i}); 
    
    for j = 1:numel(dataGroups{i})
        data = eval(dataGroups{i}{j});
        
        % Plot the first half with thinner lines
        plot3(data(StartP:EndP, 1), data(StartP:EndP, 2), data(StartP:EndP, 3), colors{j}, 'linewidth', 2)
        
        % Plot the second half with thicker lines
        plot3(data(StartP2:EndP2, 1), data(StartP2:EndP2, 2), data(StartP2:EndP2, 3), colors{j}, 'linewidth', 5)
        
        % Plot the starting points of the second half as circles
        scatter3(data(StartP2, 1), data(StartP2, 2), data(StartP2, 3), [colors{j} 'o'], 'linewidth', 1)
    end
    
    % Add a legend to the plot inside the subplot
    if i == 1
        custom_lines(1) = line(nan, nan, 'Color', colors{1}, 'LineStyle', '-', 'LineWidth', 2);
        custom_lines(2) = line(nan, nan, 'Color', colors{2}, 'LineStyle', '-', 'LineWidth', 2);
        custom_lines(3) = line(nan, nan, 'Color', colors{3}, 'LineStyle', '-', 'LineWidth', 2);
        custom_lines(4) = line(nan, nan, 'Color', colors{4}, 'LineStyle', '-', 'LineWidth', 2); 
        
        legend(custom_lines, labels, 'Location', 'northeast');
    end
    
end

%% Fig.3H & 3I Quantification of CMI differences

titles = {'DP detCMI clr', 'DP detCMI ori', 'DQ detCMI clr', 'DQ detCMI ori'};

colors = {[0.1, 0.2, 0.5], [0.8, 0.2, 0.2], [0.2, 0.7, 0.3]};

figure;

for i = 1:4
    
    switch i
        case 1
            data = DP_detCMI_clr;
        case 2
            data = DP_detCMI_ori;
        case 3
            data = DQ_detCMI_clr;
        case 4
            data = DQ_detCMI_ori;
    end
    means = mean(data);
    sems = std(data) / sqrt(size(data, 1));
    
    subplot(2, 2, i);
    hold on;
    for j = 1:3
        errorbar(j, means(j), sems(j), 'o', 'MarkerFaceColor', colors{j}, 'MarkerEdgeColor', 'k', 'Color', colors{j}, 'LineWidth', 1.5, 'MarkerSize', 8);
    end
    hold off;
    
    title(titles{i});
    xlabel('Stage');
    ylabel('CMI Value');
    xlim([0.5 3.5]);
    set(gca, 'XTick', 1:3, 'XTickLabel', {'Early Stage', 'Incorrect', 'Late Stage'});
    
    
    set(gca, 'Box', 'off');
    set(gca, 'FontSize', 10);
    grid on;
end

%% Fig.4B Time course of trial-averaged neuronal activity from an example electrode site corresponding to the three spatial conditions

% Extract time and condition data
time = Exp_Ch_Oriin(:, 1);
Oriin_Cued90 = Exp_Ch_Oriin(:, 2);
Oriin_Cued180 = Exp_Ch_Oriin(:, 3);
Orinear_Cued90 = Exp_Ch_Orinear(:, 2);
Orinear_Cued180 = Exp_Ch_Orinear(:, 3);
Orifar_Cued90 = Exp_Ch_Orifar(:, 2);
Orifar_Cued180 = Exp_Ch_Orifar(:, 3);

figure;

% First subplot: Exp_Ch_Oriin (original DMTS task)
subplot(1, 3, 1);
hold on;
plot(time, Oriin_Cued90, 'r', 'linewidth', 1.5);
plot(time, Oriin_Cued180, 'b', 'linewidth', 1.5);
set(gca, 'FontName', 'Times New Roman');
ylabel('MUA Resp');
xlabel('Time From Cue Onset (ms)');
axis([-150 1700 0.05 0.55]);
title('Oriin');
hold off;

% Second subplot: Exp_Ch_Orinear
subplot(1, 3, 2);
hold on;
plot(time, Orinear_Cued90, 'r', 'linewidth', 1.5);
plot(time, Orinear_Cued180, 'b', 'linewidth', 1.5);
set(gca, 'FontName', 'Times New Roman');
ylabel('MUA Resp');
xlabel('Time From Cue Onset (ms)');
axis([-150 1700 0.05 0.55]);
title('Orinear');
hold off;

% Third subplot: Exp_Ch_Orifar
subplot(1, 3, 3);
hold on;
plot(time, Orifar_Cued90, 'r', 'linewidth', 1.5);
plot(time, Orifar_Cued180, 'b', 'linewidth', 1.5);
set(gca, 'FontName', 'Times New Roman');
ylabel('MUA Resp');
xlabel('Time From Cue Onset (ms)');
axis([-150 1700 0.05 0.55]);
title('Orifar');
hold off;

%% Fig.4C CMIs of "near RF" and "far from RF"

% Calculate mean and standard error for each condition
DP_Stim_mean = mean(DP_Stim_near_far);
DP_Stim_sem = std(DP_Stim_near_far) / sqrt(size(DP_Stim_near_far, 1));
DP_Delay_mean = mean(DP_Delay_near_far);
DP_Delay_sem = std(DP_Delay_near_far) / sqrt(size(DP_Delay_near_far, 1));

DQ_Stim_mean = mean(DQ_Stim_near_far);
DQ_Stim_sem = std(DQ_Stim_near_far) / sqrt(size(DQ_Stim_near_far, 1));
DQ_Delay_mean = mean(DQ_Delay_near_far);
DQ_Delay_sem = std(DQ_Delay_near_far) / sqrt(size(DQ_Delay_near_far, 1));

% Plot the results
figure;

% DP Stim and Delay
subplot(1, 2, 1);
baseline_mean = mean(DP_Basline_near_far(:));
bar(1:2, [DP_Stim_mean; DP_Delay_mean]);
hold on;
errorbar([1-0.15, 1+0.15], DP_Stim_mean, DP_Stim_sem, '.', 'linewidth', 2);
errorbar([2-0.15, 2+0.15], DP_Delay_mean, DP_Delay_sem, '.', 'linewidth', 2);
plot([0.5, 2.5], [baseline_mean, baseline_mean], '--k', 'LineWidth', 1.5);
hold off;
xlabel('Condition');
ylabel('CMI');
title('DP: Mean CMI by Condition');
set(gca, 'XTick', 1:2, 'XTickLabel', {'Near', 'Far'});
legend('Stim', 'Delay');

% DQ Stim and Delay
subplot(1, 2, 2);
baseline_mean = mean(DQ_Basline_near_far(:));
bar(1:2, [DQ_Stim_mean; DQ_Delay_mean]);
hold on;
errorbar([1-0.15, 1+0.15], DQ_Stim_mean, DQ_Stim_sem, '.', 'linewidth', 2);
errorbar([2-0.15, 2+0.15], DQ_Delay_mean, DQ_Delay_sem, '.', 'linewidth', 2);
plot([0.5, 2.5], [baseline_mean, baseline_mean], '--k', 'LineWidth', 1.5);
hold off;
xlabel('Condition');
ylabel('CMI');
title('DQ: Mean CMI by Condition');
set(gca, 'XTick', 1:2, 'XTickLabel', {'Near', 'Far'});
legend('Stim', 'Delay');

%% Fig.4D Stimulus location in the "skip" task &  "stay" task

Stim_mean = mean(Stim_Attention);
Stim_sem = std(Stim_Attention) / sqrt(size(Stim_Attention, 1));
Delay_mean = mean(Delay_Attention);
Delay_sem = std(Delay_Attention) / sqrt(size(Delay_Attention, 1));

% Plot the results
figure;

% DP Stim and Delay
baseline_mean = mean(Baseline_Attention(:));
bar(1:2, [Stim_mean; Delay_mean]);
hold on;
errorbar([1-0.15, 1+0.15], Stim_mean, Stim_sem, '.', 'linewidth', 2);
errorbar([2-0.15, 2+0.15], Delay_mean, Delay_sem, '.', 'linewidth', 2);
plot([0.5, 2.5], [baseline_mean, baseline_mean], '--k', 'LineWidth', 1.5);
hold off;
xlabel('Condition');
ylabel('CMI');
title('Mean CMI by Condition');
set(gca, 'XTick', 1:2, 'XTickLabel', {'Stim', 'Delay'});
legend('Atten Outside', 'Atten Into');

%% Fig.S1 Behavioral performance for both monkeys in DMTS tasks
% Define the data groups
dataGroups = {'DP_Behavior_Summary', 'DQ_Behavior_Summary'};
labels = {'Orientation', 'Color', 'Face'};

% Create a new figure
figure

% Iterate through the data groups
for i = 1:numel(dataGroups)
    % Get the current data group
    data = eval(dataGroups{i});
    
    % Remove zeros from the data and convert to a cell array
    data_no_zeros = cell(1, size(data, 2));
    for j = 1:size(data, 2)
        data_no_zeros{j} = data(data(:, j) ~= 0, j);
    end
    
    % Find the maximum length of the data columns
    maxLength = max(cellfun(@length, data_no_zeros));
    
    % Convert the cell array to a matrix with NaN padding
    data_matrix = nan(maxLength, size(data, 2));
    for j = 1:size(data, 2)
        data_matrix(1:length(data_no_zeros{j}), j) = data_no_zeros{j};
    end
    
    subplot(1, numel(dataGroups), i)
    
    boxplot(data_matrix, 'Labels', labels)
    
    title(dataGroups{i})
    
    ylabel('Accuracy')
end

%% Fig.S2 Centers of neuronal spatial receptive fields and stimulus locations in DMTS tasks

figure

% Create the first subplot for DP_Cord data
subplot(1, 2, 1)
scatter(DP_Cord(:, 1), DP_Cord(:, 2), 'filled')
hold on
scatter(0, 0, 'r', 'x', 'LineWidth', 2)

% Calculate and plot the mean coordinates for DP_Cord
mean_DP_Cord = mean(DP_Cord, 1);
scatter(mean_DP_Cord(1), mean_DP_Cord(2), 'g', 'o', 'filled', 'LineWidth', 2)

title('DP_Cord')
xlabel('X')
ylabel('Y')
xlim([-6, 6])
ylim([-6, 6])
daspect([1 1 1]) % Set aspect ratio to 1:1

% Draw concentric circles for the first subplot
n_circles = 6;
r_max = 6;
for i = 1:n_circles
    ang = 0:0.01:2*pi;
    xp = (r_max * i / n_circles) * cos(ang);
    yp = (r_max * i / n_circles) * sin(ang);
    plot(xp, yp, 'k:');
    hold on;
end

subplot(1, 2, 2)
scatter(DQ_Cord(:, 1), DQ_Cord(:, 2), 'filled')
hold on
scatter(0, 0, 'r', 'x', 'LineWidth', 2)

mean_DQ_Cord = mean(DQ_Cord, 1);
scatter(mean_DQ_Cord(1), mean_DQ_Cord(2), 'g', 'o', 'filled', 'LineWidth', 2)

title('DQ_Cord')
xlabel('X')
ylabel('Y')
xlim([-6, 6])
ylim([-6, 6])
daspect([1 1 1]) % Set aspect ratio to 1:1

% Draw concentric circles for the second subplot
for i = 1:n_circles
    ang = 0:0.01:2*pi;
    xp = (r_max * i / n_circles) * cos(ang);
    yp = (r_max * i / n_circles) * sin(ang);
    plot(xp, yp, 'k:');
    hold on;
end

%% Fig.S3B Trial-by-trial correlation between stimulus period and delay period firing rates

experiments = {'DP0ri', 'DPclr', 'DPface', 'DQ0ri', 'DQclr', 'DQface'};

% Initialize storage for means and SEMs
means = zeros(1, 6);
sems = zeros(1, 6);

% Calculate mean and SEM for each experiment
for i = 1:6
    data = T_b_t_corr_combined(:, i);
    nonZeroData = data(data ~= 0); % Exclude zero values
    means(i) = mean(nonZeroData);
    sems(i) = std(nonZeroData) / sqrt(length(nonZeroData)); % Calculate SEM
end

% Plot point graph with error bars
figure;
colors = lines(6); % Generate 6 distinct colors
for i = 1:6
    errorbar(i, means(i), sems(i), 'o', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', 'Color', colors(i,:), 'LineWidth', 1.5, 'MarkerSize', 8);
    hold on;
end
hold off;

set(gca, 'XTick', 1:6, 'XTickLabel', experiments, 'XTickLabelRotation', 45); % Set x-axis labels and rotation
xlabel('Experiment');
ylabel('Correlation Coefficient');
title('Neuronal Correlation Across Experiments');

%% Fig.S3C Response strength during the delay period for trials with varying response strengths during the stimulus period

% Define conditions
conditions = {'DP0ri Low', 'DP0ri Medium', 'DP0ri High', ...
    'DPclr Low', 'DPclr Medium', 'DPclr High', ...
    'DPface Low', 'DPface Medium', 'DPface High', ...
    'DQ0ri Low', 'DQ0ri Medium', 'DQ0ri High', ...
    'DQclr Low', 'DQclr Medium', 'DQclr High', ...
    'DQface Low', 'DQface Medium', 'DQface High'};

% Initialize storage for means and SEMs
means = zeros(1, 18);
sems = zeros(1, 18);

% Calculate mean and SEM for each condition
for i = 1:18
    data = T_b_t_stria_combined(:, i);
    nonZeroData = data(data ~= 0); % Exclude zero values
    means(i) = mean(nonZeroData);
    sems(i) = std(nonZeroData) / sqrt(length(nonZeroData)); % Calculate SEM
end

figure;
colors = lines(18); % Generate different colors for each condition
for i = 1:18
    errorbar(i, means(i), sems(i), 'o', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'k', 'Color', colors(i,:), 'LineWidth', 1.5, 'MarkerSize', 8);
    hold on;
end
hold off;

set(gca, 'XTick', 1:18, 'XTickLabel', conditions, 'XTickLabelRotation', 45); % Set x-axis labels and rotation
xlabel('Stratified Sampling Conditions');
ylabel('Firing Rate (sp/s)');
title('Trial-by-trial Correlation between Stimulus and Delay Periods Across Conditions');
legend('hide');

%% Fig.S4A CMI values in each DMTS task

conditions = {'correct memory ori', 'incorrect ori', 'fix ori', ...
    'correct memory clr', 'incorrect clr', 'fix clr', ...
    'correct memory face', 'incorrect face', 'fix face'};

means_DP = zeros(1, 9);
sems_DP = zeros(1, 9);
for i = 1:9
    data = DP_seprated_CMI(:, i);
    nonZeroData = data(data ~= 0);
    means_DP(i) = mean(nonZeroData);
    sems_DP(i) = std(nonZeroData) / sqrt(length(nonZeroData));
end

means_DQ = zeros(1, 9);
sems_DQ = zeros(1, 9);
for i = 1:9
    data = DQ_seprated_CMI(:, i);
    nonZeroData = data(data ~= 0);
    means_DQ(i) = mean(nonZeroData);
    sems_DQ(i) = std(nonZeroData) / sqrt(length(nonZeroData));
end

figure;

% DP_seprated_CMI
subplot(1, 2, 1);
bar(means_DP, 'FaceColor', 'b');
hold on;
errorbar(1:9, means_DP, sems_DP, 'k', 'linestyle', 'none');
set(gca, 'XTick', 1:9, 'XTickLabel', conditions, 'XTickLabelRotation', 45);
ylabel('CMI Value');
title('DP Conditions');
hold off;

% DQ_seprated_CMI
subplot(1, 2, 2);
bar(means_DQ, 'FaceColor', 'r');
hold on;
errorbar(1:9, means_DQ, sems_DQ, 'k', 'linestyle', 'none');
set(gca, 'XTick', 1:9, 'XTickLabel', conditions, 'XTickLabelRotation', 45);
ylabel('CMI Value');
title('DQ Conditions');
hold off;

%% Fig.S4B CMI values calculated using hold-back dataset

conditions = {'mem', 'inc', 'fix'};

figure;

% Subplot 1: DP_hb_CMI
subplot(1, 2, 1);
% Calculate mean and standard error for each condition
means_DP = mean(DP_hb_CMI, 1);
sems_DP = std(DP_hb_CMI, 0, 1) / sqrt(size(DP_hb_CMI, 1));
% Plot bar graph with error bars
bar(means_DP, 'FaceColor', 'b'); % Use blue to represent DP
hold on;
errorbar(1:3, means_DP, sems_DP, 'k', 'linestyle', 'none');
hold off;
set(gca, 'XTick', 1:3, 'XTickLabel', conditions);
ylabel('CMI Value');
title('DP');

% Subplot 2: DQ_hb_CMI
subplot(1, 2, 2);
% Calculate mean and standard error for each condition
means_DQ = mean(DQ_hb_CMI, 1);
sems_DQ = std(DQ_hb_CMI, 0, 1) / sqrt(size(DQ_hb_CMI, 1));

bar(means_DQ, 'FaceColor', 'r'); % Use red to represent DQ
hold on;
errorbar(1:3, means_DQ, sems_DQ, 'k', 'linestyle', 'none');
hold off;
set(gca, 'XTick', 1:3, 'XTickLabel', conditions);
ylabel('CMI Value');
title('DQ');

%% Fig.S5 Anti-interference property of VWM content differentiation demonstrated in masked-DMTS

% Extract relevant data from the input matrix
time = Masked_Data_ExpCh(:, 1); % Time points
Cue135_mean = Masked_Data_ExpCh(:, 2); % Mean values for Cue135 condition
Cue135_sem = Masked_Data_ExpCh(:, 3); % Standard errors for Cue135 condition
Cue180_mean = Masked_Data_ExpCh(:, 4); % Mean values for Cue180 condition
Cue180_sem = Masked_Data_ExpCh(:, 5); % Standard errors for Cue180 condition
t_test_results = Masked_Data_ExpCh(:, 6); % t-test results

% Create a new figure window
figure;
subplot(1,4,1:3)
% Plot the mean values for Cue135 and Cue180 conditions over time
plot(time, Cue135_mean, 'r', 'LineWidth', 2);
hold on
plot(time, Cue180_mean, 'b', 'LineWidth', 2);

% Plot the standard error of the mean (SEM) as shaded areas around the mean curves
% for the Cue135 and Cue180 conditions
fill([time', fliplr(time')], [Cue135_mean' + Cue135_sem', fliplr(Cue135_mean' - Cue135_sem')], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
fill([time', fliplr(time')], [Cue180_mean' + Cue180_sem', fliplr(Cue180_mean' - Cue180_sem')], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Plot black squares to indicate significant t-test results
sig_threshold = 0.2; % Set significance threshold
VisualizeLevel = 0.42; % Set height for displaying significant markers
significantPoints = find(t_test_results >= sig_threshold); % Find significant time points
plot(time(significantPoints), repmat(VisualizeLevel, length(significantPoints), 1), 'ks', 'MarkerSize', 4);

xlim([-70, 680]); 
xlabel('Time (ms)'); 
ylabel('Mean Response'); 
title('Mean Response with SEM for Cued 135/180'); 
legend('Cued 135', 'Cued 180', 'Location', 'Best');
hold off; 

subplot(1,4,4)

timeperiods = {'ori during', 'ori after', 'clr during', 'clr after'};

means = zeros(1, 4);
sems = zeros(1, 4);

for i = 1:4
    data = Mask_CMI_summary(:, i);
    means(i) = mean(data);
    sems(i) = std(data) / sqrt(length(data));
end

errorbar(1:4, means, sems, 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b', 'LineWidth', 1.5);
set(gca, 'XTick', 1:4, 'XTickLabel', timeperiods, 'XTickLabelRotation', 45); 
xlabel('Time Segment');
ylabel('CMI Value');
title('CMI Summary Across Time');

%% Fig.S10 £¨A&B) Example neuronal activities demonstrating transformative information coding during VWM task

% Extract time and data for each condition
time = Exp_Trans_Ch(:, 1);

figure;

% Plot the data for each condition
for i = 1:6
    subplot(2, 3, i);
    plot(time, Exp_Trans_Ch(:, 2*i), 'b', 'LineWidth', 1); % Plot first sub-condition
    hold on;
    plot(time, Exp_Trans_Ch(:, 2*i + 1), 'r', 'LineWidth', 1); % Plot second sub-condition
    xlim([-150, 1700]);
    xlabel('Time (ms)');
    ylabel(['Resp of Elect Ch ', num2str(i)]);
end

%% Fig.S14B Validation of neuronal representation changes of memory content in association tasks

% Define decoder scheme labels
decoderSchemes = {'DP Early CCT', 'DP Late CCT', 'DQ Early CCT', 'DQ Late CCT'};
decoderSchemesOOT = {'DP Early OOT', 'DP Late OOT', 'DQ Early OOT', 'DQ Late OOT'};

figure;

% Subplot 1: Ac_SVM_CCT
subplot(1, 2, 1);
% Calculate the mean and SEM for each scheme
means_CCT = mean(Ac_SVM_CCT);
sems_CCT = std(Ac_SVM_CCT) / sqrt(size(Ac_SVM_CCT, 1));
% Plot point graph with error bars
errorbar(1:4, means_CCT, sems_CCT, 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b', 'LineWidth', 1.5);
set(gca, 'XTick', 1:4, 'XTickLabel', decoderSchemes, 'XTickLabelRotation', 45); % Set x-axis labels and rotation
ylabel('Accuracy');
title('CCT Decoding Schemes');

% Subplot 2: Ac_SVM_OOT
subplot(1, 2, 2);
% Calculate the mean and SEM for each scheme
means_OOT = mean(Ac_SVM_OOT);
sems_OOT = std(Ac_SVM_OOT) / sqrt(size(Ac_SVM_OOT, 1));
% Plot point graph with error bars
errorbar(1:4, means_OOT, sems_OOT, 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);
set(gca, 'XTick', 1:4, 'XTickLabel', decoderSchemesOOT, 'XTickLabelRotation', 45); % Set x-axis labels and rotation
ylabel('Accuracy');
title('OOT Decoding Schemes');

%% Fig.S15A Validation of the spatial extent of visual working memory neuronal representation in V1

% Define labels for decoder schemes
decoderSchemes = {'Stimulus Near RF Decoding Near RF', 'Delay Near RF Decoding Near RF', ...
    'Stimulus Near RF Decoding Far RF', 'Delay Near RF Decoding Far RF'};

figure;

% Ac_SVM_Sp_DP
subplot(1, 2, 1);
% Calculate the mean and SEM for each scheme
means_DP = mean(Ac_SVM_Sp_DP);
sems_DP = std(Ac_SVM_Sp_DP) / sqrt(size(Ac_SVM_Sp_DP, 1));

errorbar(1:4, means_DP, sems_DP, 'o', 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'black', 'LineWidth', 1.5);
set(gca, 'XTick', 1:4, 'XTickLabel', decoderSchemes, 'XTickLabelRotation', 45); % Set x-axis labels and rotation
ylabel('Accuracy');
title('Decoding Schemes in DP');
ylim([0 1]); % Set the y-axis range from 0 to 1

% Ac_SVM_Sp_DQ
subplot(1, 2, 2);
% Calculate the mean and SEM for each scheme
means_DQ = mean(Ac_SVM_Sp_DQ);
sems_DQ = std(Ac_SVM_Sp_DQ) / sqrt(size(Ac_SVM_Sp_DQ, 1));

errorbar(1:4, means_DQ, sems_DQ, 'o', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black', 'LineWidth', 1.5);
set(gca, 'XTick', 1:4, 'XTickLabel', decoderSchemes, 'XTickLabelRotation', 45); % Set x-axis labels and rotation
ylabel('Accuracy');
title('Decoding Schemes in DQ');
ylim([0 1]); % Set the y-axis range from 0 to 1

%% Fig.S15B Validation of the spatial extent of visual working memory neuronal representation in V1 (eye gaze part)

% Define labels for decoder schemes
decoderSchemes = {'Stimulus Near RF Decoding Near RF', 'Delay Near RF Decoding Near RF', ...
    'Stimulus Near RF Decoding Far RF', 'Delay Near RF Decoding Far RF'};

figure;

% Ac_SVM_Sp_DP
subplot(1, 2, 1);
% Calculate the mean and SEM for each scheme
means_DP = mean(Ac_SVM_Sp_DP_eye);
sems_DP = std(Ac_SVM_Sp_DP_eye) / sqrt(size(Ac_SVM_Sp_DP_eye, 1));

errorbar(1:4, means_DP, sems_DP, 'o', 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'black', 'LineWidth', 1.5);
set(gca, 'XTick', 1:4, 'XTickLabel', decoderSchemes, 'XTickLabelRotation', 45); % Set x-axis labels and rotation
ylabel('Accuracy');
title('Decoding Schemes in DP (from eye data)');
ylim([0 1]); % Set the y-axis range from 0 to 1

% Ac_SVM_Sp_DQ
subplot(1, 2, 2);
% Calculate the mean and SEM for each scheme
means_DQ = mean(Ac_SVM_Sp_DQ_eye);
sems_DQ = std(Ac_SVM_Sp_DQ_eye) / sqrt(size(Ac_SVM_Sp_DQ_eye, 1));

errorbar(1:4, means_DQ, sems_DQ, 'o', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black', 'LineWidth', 1.5);
set(gca, 'XTick', 1:4, 'XTickLabel', decoderSchemes, 'XTickLabelRotation', 45); % Set x-axis labels and rotation
ylabel('Accuracy');
title('Decoding Schemes in DQ (from eye data)');
ylim([0 1]); % Set the y-axis range from 0 to 1

%% Fig. S16 Estimation of the spatial extent of VWM content modulation

% DP data
DP_x = -1.50281;
DP_y = -1.74741;
DP_stim_pos1 = [-1, -1];
DP_stim_pos2 = [-1, 1];

DP_distance_to_stim1 = sqrt((DP_x - DP_stim_pos1(1))^2 + (DP_y - DP_stim_pos1(2))^2);
DP_distance_to_stim2 = sqrt((DP_x - DP_stim_pos2(1))^2 + (DP_y - DP_stim_pos2(2))^2);
DP_distance_to_stim3 = 0;

% input CMI...near/far/centre/near/far
DP_CMI_Stim = [0.01184, 0.00001,0.13871, 0.01184, 0.00001];
DP_CMI_Delay = [0.0107, 0.00663,0.02618, 0.0107, 0.00663];

global_min = min([min(DP_CMI_Stim), min(DP_CMI_Delay)]);
global_max = max([max(DP_CMI_Stim), max(DP_CMI_Delay)]);

% Scale to 0.0000001 to 1
scaled_min = 0.0001;
scaled_max = 1;

% Normalize function
normalize = @(x) scaled_min + (x - global_min) * (scaled_max - scaled_min) / (global_max - global_min);

% Apply normalization
DP_CMI_Stim = arrayfun(normalize, DP_CMI_Stim);
DP_CMI_Delay = arrayfun(normalize, DP_CMI_Delay);

% DQ data
DQ_x = -2.9581;
DQ_y = -0.29004;
DQ_stim_pos1 = [-2.9, 0.8];
DQ_stim_pos2 = [-2, 2];

DQ_distance_to_stim1 = sqrt((DQ_x - DQ_stim_pos1(1))^2 + (DQ_y - DQ_stim_pos1(2))^2);
DQ_distance_to_stim2 = sqrt((DQ_x - DQ_stim_pos2(1))^2 + (DQ_y - DQ_stim_pos2(2))^2);
DQ_distance_to_stim3 = 0;

% input CMI...near/far/centre/near/far
DQ_CMI_Stim = [0.04468, 9.63889E-4, 0.20508 ,0.04468, 9.63889E-4];
DQ_CMI_Delay = [0.01425, 0.00701, 0.01884, 0.01425,0.00701];

global_min_DQ = min([min(DQ_CMI_Stim), min(DQ_CMI_Delay)]);
global_max_DQ = max([max(DQ_CMI_Stim), max(DQ_CMI_Delay)]);

% Scale to 0.0000001 to 1
scaled_min = 0.0000001;
scaled_max = 1;

% Normalize function for DQ data
normalize_DQ = @(x) scaled_min + (x - global_min_DQ) * (scaled_max - scaled_min) / (global_max_DQ - global_min_DQ);

% Apply normalization to DQ datasets
DQ_CMI_Stim = arrayfun(normalize_DQ, DQ_CMI_Stim);
DQ_CMI_Delay = arrayfun(normalize_DQ, DQ_CMI_Delay);

results_DP = zeros(2, 3);
results_DQ = zeros(2, 3);
Curr_row  = 1;

for i = 1:2
    
    if i == 1
        Curr_dist1 = DP_distance_to_stim1;
        Curr_dist2 = DP_distance_to_stim2;
        Curr_dist3 = DP_distance_to_stim3;
        
        Curr_stim_CMI = DP_CMI_Stim;
        Curr_delay_CMI = DP_CMI_Delay;
    else
        Curr_dist1 = DQ_distance_to_stim1;
        Curr_dist2 = DQ_distance_to_stim2;
        Curr_dist3 = DQ_distance_to_stim3;
        
        Curr_stim_CMI = DQ_CMI_Stim;
        Curr_delay_CMI = DQ_CMI_Delay;
    end
    
    x1 = [Curr_dist1, Curr_dist2, Curr_dist3, -Curr_dist1, -Curr_dist2];
    y1 = Curr_stim_CMI;
    
    x2 = [Curr_dist1, Curr_dist2, Curr_dist3, -Curr_dist1, -Curr_dist2];
    y2 = Curr_delay_CMI;
    
    gaussianFunc = @(p, x) p(1) * exp(-((x - p(2)).^2) / (2 * p(3)^2));
    
    lb = [0, -Inf, 0];
    ub = [Inf, Inf, Inf];
    initialGuess = [0.5, 0, 2];
    options = optimoptions('lsqcurvefit', 'Display', 'off');
    
    p1 = lsqcurvefit(gaussianFunc, initialGuess, x1, y1, lb, ub, options);
    p2 = lsqcurvefit(gaussianFunc, initialGuess, x2, y2, lb, ub, options);
    
    if i == 1
        results_DP(1, :) = p1; % DP Stim
        results_DP(2, :) = p2; % DP Delay
    else
        results_DQ(1, :) = p1; % DQ Stim
        results_DQ(2, :) = p2; % DQ Delay
    end
    
    [x, y] = meshgrid(linspace(-4, 4, 1000));
    r = sqrt(x.^2 + y.^2);
    z1 = gaussianFunc(p1, r);
    z2 = gaussianFunc(p2, r);
    
    mask1 = r <= 2 * p1(3);
    mask2 = r <= 2 * p2(3);
    
    z1(~mask1) = NaN;
    z2(~mask2) = NaN;
    
    subplot(2,2,Curr_row)
    
    surf(x, y, z1, 'EdgeColor', 'none', 'FaceColor', [0.9, 0.3, 0.3], 'FaceAlpha', 0.5); hold on;
    surf(x, y, z2, 'EdgeColor', 'none', 'FaceColor', [0.3, 0.3, 0.9], 'FaceAlpha', 0.7);
    
    grid on; axis tight; view([-30, 30]);
    hold off;
    set(gcf, 'Color', 'white');
    
    Curr_row = Curr_row + 1;
    
    DP_Stim_2sigma = 2 * results_DP(1, 3); % Stim
    DP_Delay_2sigma = 2 * results_DP(2, 3); % Delay
    
    DQ_Stim_2sigma = 2 * results_DQ(1, 3); % Stim
    DQ_Delay_2sigma = 2 * results_DQ(2, 3); % Delay
    
    subplot(2,2,Curr_row)
    set(gcf, 'Color', 'white'); % Set background color to white
    
    
    % Define the x-axis range
    x = linspace(-5, 5, 1000); % Adjust the range and precision as needed
    
    % Define the Gaussian functions
    gaussian1 = p1(1) * exp(-((x - p1(2)).^2 / (2 * p1(3)^2)));
    gaussian2 = p2(1) * exp(-((x - p2(2)).^2 / (2 * p2(3)^2)));
    
    % Plotting the Gaussian functions with modified line width and colors
    plot(x, gaussian1, 'LineWidth', 2, 'Color', [1, 0.4, 0.4]); % Light red color
    hold on; % This keeps the plot open to add more data
    plot(x, gaussian2, 'LineWidth', 2, 'Color', [0.4, 0.4, 1]); % Light blue color
    
    % Function to draw vertical line
    drawVerticalLine = @(xPos, color) plot([xPos xPos], ylim, '--', 'Color', color);
    
    % Marking 2*sigma for Gaussian 1
    drawVerticalLine(p1(2) - 2*p1(3), [1, 0.4, 0.4]);
    drawVerticalLine(p1(2) + 2*p1(3), [1, 0.4, 0.4]);
    
    % Marking 2*sigma for Gaussian 2
    drawVerticalLine(p2(2) - 2*p2(3), [0.4, 0.4, 1]);
    drawVerticalLine(p2(2) + 2*p2(3), [0.4, 0.4, 1]);
    
    % Plotting your data as a scatter plot
    scatter(x1, y1, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r'); % First set of data points
    scatter(x2, y2, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b'); % Second set of data points
    
    % Enhancing the plot
    xlabel('Distance from RF center to stim');
    ylabel('Estimated CMI');
    legend('Stim', 'Delay');
    
    hold off; % Release the plot
    Curr_row = Curr_row + 1;
    
end
