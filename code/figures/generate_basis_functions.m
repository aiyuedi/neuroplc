%% Figure: C^2-BV Architecture Basis Function Comparison
%  Visualize the 5 SVNN-compliant activation basis functions on [-3,3]
%  with their M2 (curvature bound) values annotated

clc; clear; close all;

output_dir = 'D:/neuroplc-paper/paper/figures';

rng(42);
xs = linspace(-3, 3, 500);
N_LUT = 15;
grid_pts = linspace(-3, 3, N_LUT);

fig = figure('Position', [50, 50, 1400, 900]);

% ── 1. B-spline KAN (cubic) ──
subplot(2,3,1);
% Simulate a cubic B-spline: sum of piecewise cubics
% Use a random combination of basis functions near a typical KAN edge
knots = linspace(-3.5, 3.5, 12);
c = [0.2, -0.5, 0.8, -0.3, 0.6, -0.1, 0.4, -0.2];
phi_bspline = zeros(size(xs));
for i = 1:length(xs)
    x = xs(i);
    val = 0;
    for j = 1:length(c)
        % B-spline basis: tent-like cubic approximation
        t = (x - knots(j)) / (knots(2)-knots(1));
        if abs(t) < 2
            val = val + c(j) * max(0, (1-abs(t)/2)^3);
        end
    end
    phi_bspline(i) = val;
end
plot(xs, phi_bspline, 'b-', 'LineWidth', 1.5);
hold on;
plot(grid_pts, interp1(xs, phi_bspline, grid_pts), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
M2_bs = max(abs(diff(diff(phi_bspline)) / (xs(2)-xs(1))^2));
title(sprintf('B-spline KAN\nM_2 = %.3f', M2_bs), 'FontSize', 11);
xlabel('x'); ylabel('\phi(x)'); grid on; xlim([-3 3]);

% ── 2. FourierKAN ──
subplot(2,3,2);
K = 6; omega = 0.4;
ck = randn(K,1)*0.3; dk = randn(K,1)*0.3;
phi_fourier = zeros(size(xs));
for k = 1:K
    phi_fourier = phi_fourier + ck(k)*sin(k*omega*xs) + dk(k)*cos(k*omega*xs);
end
plot(xs, phi_fourier, 'r-', 'LineWidth', 1.5);
hold on;
plot(grid_pts, interp1(xs, phi_fourier, grid_pts), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
M2_f = omega^2 * sum((1:K)'.^2 .* (abs(ck)+abs(dk)));
title(sprintf('FourierKAN (K=%d)\nM_2 = %.3f', K, M2_f), 'FontSize', 11);
xlabel('x'); ylabel('\phi(x)'); grid on; xlim([-3 3]);

% ── 3. WaveletKAN (Mexican hat) ──
subplot(2,3,3);
a_scales = [0.4, 0.8, 1.6, 3.0];
b_shifts = [-1.5, -0.5, 0.5, 1.5];
cw = randn(4,1)*0.4;
phi_wavelet = zeros(size(xs));
for j = 1:4
    t = (xs - b_shifts(j)) / a_scales(j);
    psi = (2/sqrt(3))*pi^(-1/4) * (1 - t.^2) .* exp(-t.^2/2);
    phi_wavelet = phi_wavelet + cw(j) * psi;
end
plot(xs, phi_wavelet, 'g-', 'LineWidth', 1.5);
hold on;
plot(grid_pts, interp1(xs, phi_wavelet, grid_pts), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
M2_w = max(abs(cw)) / min(a_scales)^2 * 2.602;
title(sprintf('WaveletKAN (Mexican hat)\nM_2 = %.3f', M2_w), 'FontSize', 11);
xlabel('x'); ylabel('\phi(x)'); grid on; xlim([-3 3]);

% ── 4. ChebyKAN ──
subplot(2,3,4);
deg = 5;
cc = randn(deg+1,1)*0.3; cc(1)=0;
% Chebyshev polynomials via recurrence
T = zeros(length(xs), deg+1);
T(:,1) = 1;
if deg >= 1, T(:,2) = xs; end
for n = 3:deg+1
    T(:,n) = 2*xs'.*T(:,n-1) - T(:,n-2);
end
phi_cheby = T * cc;
plot(xs, phi_cheby, 'm-', 'LineWidth', 1.5);
hold on;
plot(grid_pts, interp1(xs, phi_cheby, grid_pts), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
M2_c = sum((0:deg)'.^4 .* abs(cc));
title(sprintf('ChebyKAN (deg=%d)\nM_2 = %.3f', deg, M2_c), 'FontSize', 11);
xlabel('x'); ylabel('\phi(x)'); grid on; xlim([-3 3]);

% ── 5. RBF-KAN (Gaussian) ──
subplot(2,3,5);
sigma_rbf = 0.6;
crbf = randn(1)*0.8;
phi_rbf = crbf * exp(-xs.^2 / sigma_rbf^2);
plot(xs, phi_rbf, 'c-', 'LineWidth', 1.5);
hold on;
plot(grid_pts, interp1(xs, phi_rbf, grid_pts), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', 'k');
M2_r = 2*abs(crbf) / sigma_rbf^2;
title(sprintf('RBF-KAN (Gauss, \x03C3=%.1f)\nM_2 = %.3f', sigma_rbf, M2_r), 'FontSize', 11);
xlabel('x'); ylabel('\phi(x)'); grid on; xlim([-3 3]);

% ── 6. ALL LUT points overlay ──
subplot(2,3,6);
plot(xs, phi_bspline, 'b-', 'LineWidth', 1.2); hold on;
plot(xs, phi_fourier, 'r-', 'LineWidth', 1.2);
plot(xs, phi_wavelet, 'g-', 'LineWidth', 1.2);
plot(xs, phi_cheby, 'm-', 'LineWidth', 1.2);
plot(xs, phi_rbf, 'c-', 'LineWidth', 1.2);
plot(grid_pts, zeros(size(grid_pts)), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k');
xlabel('x'); ylabel('\phi(x)'); grid on; xlim([-3 3]);
title(sprintf('All C^2-BV Bases + %d LUT Grid Points', N_LUT), 'FontSize', 11);
legend({'B-spline','Fourier','Wavelet','Chebyshev','RBF','LUT pts'}, ...
    'Location', 'best', 'FontSize', 8);

sgtitle('C^2-BV Architecture Family: Activation Basis Functions with LUT Discretization', ...
    'FontSize', 14, 'FontWeight', 'bold');

saveas(fig, fullfile(output_dir, 'fig_c2bv_basis_functions.pdf'));
saveas(fig, fullfile(output_dir, 'fig_c2bv_basis_functions.png'));
close(fig);
fprintf('[DONE] fig_c2bv_basis_functions.pdf generated\n');

%% ── Also generate WCET breakdown bar chart ──
fig2 = figure('Position', [100, 100, 700, 450]);

categories = {'LUT L0\n(448 edges)', 'LUT L1\n(64 edges)', 'MatMul\nL0+L1', 'Softmax', 'Overhead'};
times_us = [16442, 2349, 3702, 109, 72];  % From Theorem 10 analysis
total = sum(times_us);

b = bar(1:5, times_us);
b.FaceColor = 'flat';
b.CData = [0.2 0.4 0.8; 0.2 0.4 0.8; 0.2 0.8 0.4; 0.9 0.6 0.2; 0.6 0.6 0.6];
set(gca, 'XTickLabel', categories, 'FontSize', 10);
ylabel('Execution Time (\mus)', 'FontSize', 12);
title(sprintf('WCET Breakdown: KAN [28,16,4] on S7-1200\nTotal = %.1f ms (%s%% of 100 ms scan cycle)', ...
    total/1000, sprintf('%.1f', total/1000/100*100)), 'FontSize', 13);
grid on;

for i = 1:5
    pct = times_us(i)/total*100;
    text(i, times_us(i)+200, sprintf('%.0f \\mus\n(%.1f%%)', times_us(i), pct), ...
        'HorizontalAlign', 'center', 'FontSize', 9, 'FontWeight', 'bold');
end

saveas(fig2, fullfile(output_dir, 'fig_wcet_breakdown.pdf'));
saveas(fig2, fullfile(output_dir, 'fig_wcet_breakdown.png'));
close(fig2);
fprintf('[DONE] fig_wcet_breakdown.pdf generated\n');

%% ── Theorem dependency map (text-based figure) ──
% Not a plot, but a structured table that shows theorem dependencies
fprintf('[DONE] All figures generated in %s\n', output_dir);
