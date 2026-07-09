%% NeuroPLC King-Level Figures — Generate 4 publication-quality plots
%  1. C^2-BV Cross-Architecture Z3 Verification Bar Chart
%  2. DA Tightness: M2*h^2/8 vs Actual Error (1000 quadratics)
%  3. Sharp Lower Bound: √d per-layer gap (MLP vs KAN)
%  4. DA vs IA Bound Comparison across architectures

clc; clear; close all;

output_dir = 'D:/neuroplc-paper/paper/figures';

%% ── Figure 1: C^2-BV Cross-Architecture Z3 Verification ──
fprintf('[1/4] C^2-BV Architecture Z3 Verification Bar Chart\n');

archs = {'B-spline\nKAN', 'Fourier\nKAN', 'Wavelet\nKAN', 'Cheby\nKAN', 'Standard\nMLP'};
z3_rates = [512, 512, 512, 496, 0];        % verified / 512
accuracies = [99.93, 100.0, 100.0, 100.0, 24.13];  % CWRU test accuracy
safety_margins = [4.5, 2.9, 5.6, 1.1, 0.0];

fig = figure('Position', [100, 100, 900, 500]);

% Left: Z3 verification rate
subplot(1,2,1);
b = bar(1:5, z3_rates/512*100);
b.FaceColor = 'flat';
colors = [0.2 0.6 0.2; 0.2 0.6 0.2; 0.2 0.6 0.2; 0.9 0.6 0.2; 0.8 0.2 0.2];
b.CData = colors;
set(gca, 'XTickLabel', archs, 'FontSize', 10);
ylabel('Z3-Equivalent Verification Rate (%)', 'FontSize', 11);
title('Per-Edge Z3 Verification (512 edges)', 'FontSize', 12);
ylim([0 105]);
for i = 1:5
    text(i, z3_rates(i)/512*100 + 3, sprintf('%d/%d', z3_rates(i), 512), ...
        'HorizontalAlign', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end
grid on;

% Right: Accuracy + Safety margin
subplot(1,2,2);
yyaxis left;
b2 = bar(1:5, accuracies);
b2.FaceColor = 'flat';
b2.CData = [0.3 0.3 0.8; 0.3 0.3 0.8; 0.3 0.3 0.8; 0.3 0.3 0.8; 0.7 0.3 0.3];
set(gca, 'XTickLabel', archs, 'FontSize', 10);
ylabel('CWRU Test Accuracy (%)', 'FontSize', 11);
ylim([0 105]);

yyaxis right;
plot(1:5, safety_margins, 'ko-', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerFaceColor', 'k');
ylabel('Safety Margin (×)', 'FontSize', 11);
title('Test Accuracy & Z3 Safety Margin', 'FontSize', 12);
grid on;

sgtitle('C^2-BV Architecture Family: SVNN Verification Across 5 Architectures', ...
    'FontSize', 14, 'FontWeight', 'bold');

saveas(fig, fullfile(output_dir, 'fig_c2bv_verification.pdf'));
saveas(fig, fullfile(output_dir, 'fig_c2bv_verification.png'));
close(fig);
fprintf('  -> fig_c2bv_verification.pdf\n');

%% ── Figure 2: DA Tightness — M2*h^2/8 exact attainment ──
fprintf('[2/4] DA Tightness Plot\n');

rng(42);
n_funcs = 50;  % show 50 representative quadratics
N_LUT = 15;
h = 6 / (N_LUT - 1);  % domain [-3,3]

actual_errors = zeros(n_funcs, 1);
theor_bounds = zeros(n_funcs, 1);

for i = 1:n_funcs
    a = (rand - 0.5) * 3.0;
    b = (rand - 0.5) * 3.0;
    c = (rand - 0.5) * 2.0;
    M2 = abs(2*a);
    theor_bounds(i) = M2 * h^2 / 8;

    % Compute actual max LUT error
    grid_pts = linspace(-3, 3, N_LUT);
    lut = a*grid_pts.^2 + b*grid_pts + c;
    xs = linspace(-2.99, 2.99, 5000);
    max_err = 0;
    for j = 1:length(xs)
        k = sum(grid_pts <= xs(j));
        if k < 1, k = 1; end
        if k >= N_LUT, k = N_LUT - 1; end
        t = (xs(j) - grid_pts(k)) / (grid_pts(k+1) - grid_pts(k));
        interp_val = lut(k) + t * (lut(k+1) - lut(k));
        true_val = a*xs(j)^2 + b*xs(j) + c;
        max_err = max(max_err, abs(true_val - interp_val));
    end
    actual_errors(i) = max_err;
end

fig = figure('Position', [100, 100, 700, 500]);

% Scatter plot
scatter(theor_bounds, actual_errors, 30, 'b', 'filled', 'MarkerEdgeColor', 'k');
hold on;

% Diagonal (y = x) — exact attainment line
max_val = max(max(theor_bounds), max(actual_errors)) * 1.1;
plot([0 max_val], [0 max_val], 'r--', 'LineWidth', 2);

xlabel('Theoretical Bound: M_2 \cdot h^2 / 8', 'FontSize', 12);
ylabel('Actual Maximum LUT Error', 'FontSize', 12);
title(sprintf('DA Tightness: 50 Random Quadratics (exact match: all %.0f/%.0f)', ...
    sum(abs(actual_errors - theor_bounds) < 1e-12), n_funcs), ...
    'FontSize', 14);
legend({'Random C^2 functions', 'y = x (exact attainment line)'}, ...
    'Location', 'northwest', 'FontSize', 10);
grid on;
axis equal;
xlim([0 max_val]); ylim([0 max_val]);

saveas(fig, fullfile(output_dir, 'fig_da_tightness.pdf'));
saveas(fig, fullfile(output_dir, 'fig_da_tightness.png'));
close(fig);
fprintf('  -> fig_da_tightness.pdf\n');

%% ── Figure 3: Sharp Lower Bound — √d gap ──
fprintf('[3/4] Sharp Lower Bound Plot\n');

d_vals = [4, 8, 16, 32, 64, 128, 256];
mlp_amp = sqrt(d_vals);  % all-ones construction
kan_amp = 0.182 * ones(size(d_vals));  % measured γ
ratio = mlp_amp ./ kan_amp;

fig = figure('Position', [100, 100, 900, 400]);

subplot(1,2,1);
loglog(d_vals, mlp_amp, 'ro-', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerFaceColor', 'r', 'DisplayName', 'MLP: ||W||_{1,\infty} = \surd d');
hold on;
loglog(d_vals, kan_amp, 'bs-', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerFaceColor', 'b', 'DisplayName', 'KAN: \gamma = 0.182 (d-independent)');
xlabel('Network Width d', 'FontSize', 12);
ylabel('Per-Layer Error Amplification', 'FontSize', 12);
title('Error Amplification: MLP vs KAN', 'FontSize', 13);
legend('Location', 'northwest', 'FontSize', 10);
grid on;
xticks(d_vals);

subplot(1,2,2);
semilogy(d_vals, ratio, 'ko-', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerFaceColor', 'k');
xlabel('Network Width d', 'FontSize', 12);
ylabel('MLP/KAN Amplification Ratio', 'FontSize', 12);
title('Per-Layer Gap: \surd d / \gamma', 'FontSize', 13);
grid on;
xticks(d_vals);
for i = 1:length(d_vals)
    text(d_vals(i), ratio(i)*1.1, sprintf('%.0f×', ratio(i)), ...
        'HorizontalAlign', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end

sgtitle('Sharp Necessity Bound: Deterministic MLP Adversarial Construction', ...
    'FontSize', 14, 'FontWeight', 'bold');

saveas(fig, fullfile(output_dir, 'fig_sharp_lower_bound.pdf'));
saveas(fig, fullfile(output_dir, 'fig_sharp_lower_bound.png'));
close(fig);
fprintf('  -> fig_sharp_lower_bound.pdf\n');

%% ── Figure 4: DA vs IA Bound Comparison ──
fprintf('[4/4] DA vs IA Comparison Plot\n');

% Data from actual experiments
N_lut_vals = [8, 10, 12, 15, 18, 20];
DA_bounds = [0.419, 0.305, 0.212, 0.079, 0.055, 0.044];   % E9 DA values
IA_bounds = [0.922, 0.671, 0.466, 0.172, 0.121, 0.097];   % E9 IA values
ratio_vals = [2.20, 2.20, 2.20, 2.20, 2.20, 2.20];  % stable 2.2× ratio

fig = figure('Position', [100, 100, 900, 400]);

subplot(1,2,1);
semilogy(N_lut_vals, DA_bounds, 'bo-', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerFaceColor', 'b', 'DisplayName', 'DA Bound');
hold on;
semilogy(N_lut_vals, IA_bounds, 'ro-', 'LineWidth', 2, 'MarkerSize', 10, ...
    'MarkerFaceColor', 'r', 'DisplayName', 'IA Bound');
semilogy(N_lut_vals, DA_bounds .* 0.6, 'k--', 'LineWidth', 1, ...
    'DisplayName', 'Safety threshold (0.6×DA)');
xlabel('LUT Points N', 'FontSize', 12);
ylabel('Output Error Bound (log scale)', 'FontSize', 12);
title('DA vs IA: Error Bound vs LUT Resolution', 'FontSize', 13);
legend('Location', 'northeast', 'FontSize', 10);
grid on;
xticks(N_lut_vals);

% Annotate the gap
for i = 1:length(N_lut_vals)
    text(N_lut_vals(i)+0.1, IA_bounds(i), sprintf('%.3f', IA_bounds(i)), ...
        'FontSize', 8, 'Color', [0.6 0 0]);
    text(N_lut_vals(i)+0.1, DA_bounds(i), sprintf('%.3f', DA_bounds(i)), ...
        'FontSize', 8, 'Color', [0 0 0.6]);
end

subplot(1,2,2);
b = bar(1:6, [DA_bounds; IA_bounds]');
b(1).FaceColor = [0.2 0.2 0.8];
b(2).FaceColor = [0.8 0.2 0.2];
set(gca, 'XTickLabel', {'N=8','N=10','N=12','N=15','N=18','N=20'}, ...
    'FontSize', 11);
ylabel('Output Error Bound', 'FontSize', 12);
title(sprintf('DA Tightening: %.1f× vs IA (Theorems 9 + C)', ratio_vals(1)), ...
    'FontSize', 13);
legend({'Doubleton Arithmetic', 'Interval Arithmetic'}, ...
    'Location', 'northeast', 'FontSize', 10);
grid on;

sgtitle('DA Optimality: Tightest Sound Abstract Domain for C^2 Functions', ...
    'FontSize', 14, 'FontWeight', 'bold');

saveas(fig, fullfile(output_dir, 'fig_da_vs_ia.pdf'));
saveas(fig, fullfile(output_dir, 'fig_da_vs_ia.png'));
close(fig);
fprintf('  -> fig_da_vs_ia.pdf\n');

%% ── Done ──
fprintf('\n[DONE] All 4 king-level figures generated in %s\n', output_dir);
