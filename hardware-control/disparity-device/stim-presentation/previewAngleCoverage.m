function previewAngleCoverage(left_slope, left_intercept, right_slope, right_intercept, opts)
% Plot the deg->pixel mapping for both eyes (no hardware), to eyeball screen
% coverage and disparity before running teensyStimulusAngle.
%
% Same fit convention as teensyStimulusAngle:
%   right_px = right_slope*deg + right_intercept                 (valid in [0, W])
%   left_px  = left_slope*(deg + disparity_deg) + left_intercept (valid in [-W, 0])
%
% Usage:
%   previewAngleCoverage(0.64, -44.7, 0.88, 64.4)
%   previewAngleCoverage(0.64, -44.7, 0.88, 64.4, 'disparities', [-10 0 10])

arguments
    left_slope      (1,1) double
    left_intercept  (1,1) double
    right_slope     (1,1) double
    right_intercept (1,1) double
    opts.start_deg   (1,1) double = -30
    opts.end_deg     (1,1) double = 60
    opts.disparities (1,:) double = 0     % left-eye disparity offsets (degrees)
    opts.W           (1,1) double = 240
    opts.n           (1,1) double = 200   % sampling density for the curves
end

deg      = linspace(opts.start_deg, opts.end_deg, opts.n);
right_px = right_slope * deg + right_intercept;

figure('Name', 'Angle -> pixel coverage', 'NumberTitle', 'off'); hold on;

% Valid screen bands: right eye [0, W], left eye [-W, 0]
yl = [-opts.W, opts.W];
fill([opts.start_deg opts.end_deg opts.end_deg opts.start_deg], [0 0 opts.W opts.W], ...
     [0.85 0.9 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.5);      % right screen band
fill([opts.start_deg opts.end_deg opts.end_deg opts.start_deg], [-opts.W -opts.W 0 0], ...
     [1.0 0.88 0.88], 'EdgeColor', 'none', 'FaceAlpha', 0.5);     % left screen band
yline(0, 'k--', 'LineWidth', 1);   % inner (nasal) edge shared by both screens

% Right eye (shared across disparities)
plot(deg, right_px, 'b-', 'LineWidth', 2, 'DisplayName', 'right eye');
mark_out_of_range(deg, right_px, 0, opts.W);

% Left eye, one curve per disparity condition
cmap = spring(max(numel(opts.disparities), 2));
for k = 1:numel(opts.disparities)
    D = opts.disparities(k);
    left_px = left_slope * (deg + D) + left_intercept;
    plot(deg, left_px, '-', 'Color', cmap(k,:), 'LineWidth', 2, ...
         'DisplayName', sprintf('left eye (disp=%+.1f)', D));
    mark_out_of_range(deg, left_px, -opts.W, 0);
end

xlabel('Visual angle (deg)'); ylabel('Screen pixel');
title('Per-eye deg\rightarrowpixel coverage (bands = valid screen range)');
ylim(yl); legend('Location', 'best', 'Box', 'off'); grid on; box off;

% Report endpoints and any clipping
fprintf('right eye: %.1f -> %.1f px  (valid [0, %d])\n', ...
    right_px(1), right_px(end), opts.W);
for k = 1:numel(opts.disparities)
    D = opts.disparities(k);
    lp = left_slope * ([opts.start_deg opts.end_deg] + D) + left_intercept;
    fprintf('left eye  (disp=%+.1f): %.1f -> %.1f px  (valid [%d, 0])\n', ...
        D, lp(1), lp(2), -opts.W);
end

end

%% Highlight any part of a curve that falls outside its screen band
function mark_out_of_range(deg, px, lo, hi)
    oob = px < lo | px > hi;
    if any(oob)
        plot(deg(oob), px(oob), 'rx', 'MarkerSize', 5, 'HandleVisibility', 'off');
    end
end
