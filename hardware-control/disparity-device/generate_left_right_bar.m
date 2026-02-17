function generate_left_right_bar(opts)
% Generate left-to-right and right-to-left checkerboard bar sweep MP4s.
%
% Usage:
%   generate_left_right_bar()
%   generate_left_right_bar('duration', 6, 'fps', 10)
%
% Optional:
%   duration   - Sweep duration in seconds (default: 6)
%   fps        - Frame rate (default: 10)
%   bar_width  - Bar width in pixels (default: 26)
%   n_checks_y - Number of checkerboard squares vertically (default: 5)
%   screen_size - Display size in pixels (default: 240)

arguments
    opts.duration (1,1) double = 12
    opts.fps (1,1) double = 10
    opts.bar_width (1,1) double = 26
    opts.n_checks_y (1,1) double = 5
    opts.screen_size (1,1) double = 240
end

S = opts.screen_size;
bw = opts.bar_width;
num_frames = round(opts.duration * opts.fps);
check_w = bw / 2;          % 13px wide squares
check_h = S / opts.n_checks_y; % 48px tall squares
gray = uint8(127);

% Build the checkerboard bar column pattern (S x bw)
[cx, cy] = meshgrid(1:bw, 1:S);
checker = uint8(255 * xor(mod(floor((cx-1)/check_w),2), mod(floor((cy-1)/check_h),2)));
% checker is 0 or 255, alternating squares

names = {'bar_left_to_right.mp4', 'bar_right_to_left.mp4'};
labels = {'left-to-right', 'right-to-left'};

for d = 1:2
    out_path = fullfile(pwd, names{d});
    vid = VideoWriter(out_path, 'MPEG-4');
    vid.FrameRate = opts.fps;
    vid.Quality = 95;
    open(vid);

    fprintf('Writing %s (%d frames) ...\n', labels{d}, num_frames);
    for f = 1:num_frames
        % Bar center position: sweeps from 1 to S
        if d == 1
            t = (f - 1) / max(1, num_frames - 1);
        else
            t = 1 - (f - 1) / max(1, num_frames - 1);
        end
        center_x = round(1 + t * (S - 1));

        % Bar column range
        x1 = max(1, center_x - floor(bw/2));
        x2 = min(S, center_x + floor(bw/2) - 1);

        % Gray background
        frame = repmat(gray, [S, S]);

        % Place visible portion of checkerboard bar
        bar_cols = x1:x2;
        checker_cols = bar_cols - (center_x - floor(bw/2)) + 1;
        checker_cols = max(1, min(bw, checker_cols));
        frame(:, bar_cols) = checker(:, checker_cols);

        frame_rgb = cat(3, frame, frame, frame);
        writeVideo(vid, frame_rgb);
    end

    close(vid);
    fprintf('  Saved: %s\n', out_path);
end

fprintf('\nDone. 2 videos @ %dx%d, %d fps, %ds duration\n', S, S, opts.fps, opts.duration);

end
