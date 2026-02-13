% Plot tuning results (Top-K) similar to Phase 4 plots
% Usage in MATLAB: run plot_tuning_results

inFile = fullfile(pwd, 'outputs', 'tuning_results.csv');
outDir = fullfile(pwd, 'outputs');
if ~exist(outDir, 'dir'); mkdir(outDir); end

T = readtable(inFile);

% Ensure numeric columns are parsed
numCols = {'val_score','val_success_rate','val_safe_success_rate','val_collision_rate',...
    'val_dist_mean','val_yaw_mean','val_steps_mean'};
for i = 1:numel(numCols)
    if iscell(T.(numCols{i}))
        T.(numCols{i}) = str2double(T.(numCols{i}));
    end
end

% Sort by validation score
[~, idx] = sort(T.val_score, 'descend');
T = T(idx, :);

K = min(20, height(T));
Tk = T(1:K, :);
labels = strcat('trial_', string(Tk.trial));

figure('Color','w','Position',[100 100 1200 700]);

subplot(2,2,1);
bar(Tk.val_score); grid on;
xticks(1:K); xticklabels(labels); xtickangle(45);
set(gca,'FontSize',8);
title('Top-K Validation Score');

subplot(2,2,2);
bar([Tk.val_success_rate, Tk.val_safe_success_rate]); grid on;
xticks(1:K); xticklabels(labels); xtickangle(45);
set(gca,'FontSize',8);
title('Success vs Safe Success');
legend({'success','safe success'}, 'Location','best');

subplot(2,2,3);
bar(Tk.val_steps_mean); grid on;
xticks(1:K); xticklabels(labels); xtickangle(45);
set(gca,'FontSize',8);
title('Mean Steps (Validation)');

subplot(2,2,4);
bar(Tk.val_collision_rate); grid on;
xticks(1:K); xticklabels(labels); xtickangle(45);
set(gca,'FontSize',8);
title('Collision Rate (Validation)');

sgtitle('Phase 5 Tuning Results (Top-K)');

outFile = fullfile(outDir, 'tuning_results_topk.png');
exportgraphics(gcf, outFile, 'Resolution', 150);

fprintf('Saved %s\n', outFile);
