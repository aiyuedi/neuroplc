% Fixed Part 5: Correct IA vs DA numerical comparison
rng(42);
n_trials = 500;
da_over_arr = zeros(n_trials, 1);
ia_over_arr = zeros(n_trials, 1);
true_rad_arr = zeros(n_trials, 1);

for i = 1:n_trials
    a3v = (rand - 0.5) * 2.0;
    a2v = (rand - 0.5) * 3.0;
    a1v = (rand - 0.5) * 4.0;
    a0v = (rand - 0.5) * 1.0;

    x0v = (rand - 0.5) * 0.6;
    rv = 0.02 + rand * 0.98;

    % DA bound
    C1v = rv * (a1v + 2*a2v*x0v + 3*a3v*x0v^2);
    C2v = rv^2 * (a2v + 3*a3v*x0v);
    C3v = a3v * rv^3;
    da_bound = abs(C1v) + abs(C2v) + abs(C3v);

    % True range
    xs = linspace(x0v - rv, x0v + rv, 1000);
    ys = a3v*xs.^3 + a2v*xs.^2 + a1v*xs + a0v;
    true_radius = (max(ys) - min(ys)) / 2;

    % IA bound: naive interval arithmetic
    X_lo = x0v - rv;
    X_hi = x0v + rv;

    % Linear term
    if a1v >= 0
        ia1_lo = a1v * X_lo; ia1_hi = a1v * X_hi;
    else
        ia1_lo = a1v * X_hi; ia1_hi = a1v * X_lo;
    end

    % Quadratic: X^2 = X * X (IA multiplication)
    sq_lo = min([X_lo^2, X_hi^2, X_lo*X_hi]);
    sq_hi = max([X_lo^2, X_hi^2, X_lo*X_hi]);
    if X_lo <= 0 && X_hi >= 0
        sq_lo = 0;
    end
    if a2v >= 0
        ia2_lo = a2v * sq_lo; ia2_hi = a2v * sq_hi;
    else
        ia2_lo = a2v * sq_hi; ia2_hi = a2v * sq_lo;
    end

    % Cubic: X^3 = X^2 * X (IA multiplication = wrapping effect)
    cu_candidates = [X_lo^3, X_hi^3, sq_lo*X_lo, sq_lo*X_hi, sq_hi*X_lo, sq_hi*X_hi];
    cu_lo = min(cu_candidates);
    cu_hi = max(cu_candidates);
    if a3v >= 0
        ia3_lo = a3v * cu_lo; ia3_hi = a3v * cu_hi;
    else
        ia3_lo = a3v * cu_hi; ia3_hi = a3v * cu_lo;
    end

    % IA total
    ia_tot_lo = a0v + ia1_lo + ia2_lo + ia3_lo;
    ia_tot_hi = a0v + ia1_hi + ia2_hi + ia3_hi;
    ia_radius = (ia_tot_hi - ia_tot_lo) / 2;

    da_over_arr(i) = max(0, da_bound - true_radius);
    ia_over_arr(i) = max(0, ia_radius - true_radius);
    true_rad_arr(i) = true_radius;
end

valid = ia_over_arr > 1e-12;
ratios_valid = da_over_arr(valid) ./ ia_over_arr(valid);

fprintf('FIXED Part 5: Naive IA vs DA overestimation\n');
fprintf('  n = %d random cubic polynomials\n', n_trials);
fprintf('  Mean DA overestimation: %.6f\n', mean(da_over_arr));
fprintf('  Mean IA overestimation: %.6f\n', mean(ia_over_arr));
fprintf('  Mean DA/IA ratio: %.4f\n', mean(ratios_valid));
fprintf('  DA tighter in %d/%d (%.1f%%)\n', ...
    sum(da_over_arr < ia_over_arr), n_trials, ...
    100*mean(da_over_arr < ia_over_arr));

% O(r^2) scaling verification
radii = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8];
fprintf('\nO(r^2) scaling verification:\n');
for j = 1:length(radii)
    r_test = radii(j);
    da_vals = zeros(100, 1);
    for k = 1:100
        a3v = (rand-0.5)*2; a2v = (rand-0.5)*3;
        a1v = (rand-0.5)*4; a0v = (rand-0.5);
        x0v = (rand-0.5)*0.6;
        C1v = r_test*(a1v+2*a2v*x0v+3*a3v*x0v^2);
        C2v = r_test^2*(a2v+3*a3v*x0v);
        C3v = a3v*r_test^3;
        da_bound = abs(C1v)+abs(C2v)+abs(C3v);
        xs = linspace(x0v-r_test, x0v+r_test, 500);
        ys = a3v*xs.^3+a2v*xs.^2+a1v*xs+a0v;
        true_r = (max(ys)-min(ys))/2;
        da_vals(k) = max(0, da_bound - true_r);
    end
    fprintf('  r=%.2f: mean DA overest = %.6f\n', r_test, mean(da_vals));
end
