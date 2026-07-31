clc; close all;
root = 'D:/neuroplc-paper/paper/figures/final';
files = dir(fullfile(root, '*.png'));
report = strings(numel(files)+1, 4);
report(1,:) = ["file", "gray_std", "gray_p05", "gray_p95"];
for k = 1:numel(files)
    fn = fullfile(files(k).folder, files(k).name);
    im = im2double(imread(fn));
    if size(im,3) == 4
        im = im(:,:,1:3);
    end
    if size(im,3) == 3
        g = rgb2gray(im);
    else
        g = im;
    end
    vals = g(:);
    report(k+1,:) = [string(files(k).name), sprintf('%.4f', std(vals)), sprintf('%.4f', prctile(vals,5)), sprintf('%.4f', prctile(vals,95))];
end
out = fullfile(root, 'grayscale_report.csv');
writematrix(report, out);
disp(report);
