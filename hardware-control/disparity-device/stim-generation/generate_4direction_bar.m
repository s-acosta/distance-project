function generate_4direction_bar(opts)
% Generate 4-direction checkerboard bar sweep MP4s for sign mapping.
%
% Creates bar sweep videos in all 4 cardinal directions for use with
% teensySignMapStim.m and the SignMapping analysis pipeline.
%
% Usage:
%   generate_4direction_bar()
%   generate_4direction_bar('duration', 12, 'fps', 10)
%
% Optional:
%   duration   - Sweep duration in seconds (default: 12)
%   fps        - Frame rate (default: 10)
%   bar_width  - Bar width in pixels (default: 26)
%   n_checks_y - Number of checkerboard squares along bar length (default: 5)
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
check_w = bw / 2;
check_h = S / opts.n_checks_y;
gray = uint8(127);

% Build vertical checkerboard bar (S x bw) for horizontal sweeps
[cx, cy] = meshgrid(1:bw, 1:S);
checker_v = uint8(255 * xor(mod(floor((cx-1)/check_w),2), mod(floor((cy-1)/check_h),2)));

% Build horizontal checkerboard bar (bw x S) for vertical sweeps
[cx2, cy2] = meshgrid(1:S, 1:bw);
checker_h = uint8(255 * xor(mod(floor((cx2-1)/check_h),2), mod(floor((cy2-1)/check_w),2)));

names = {'bar_left_to_right.mp4', 'bar_right_to_left.mp4', ...
         'bar_bottom_to_top.mp4', 'bar_top_to_bottom.mp4'};
labels = {'left-to-right', 'right-to-left', 'bottom-to-top', 'top-to-bottom'};

script_dir = fileparts(mfilename('fullpath'));
stim_video_dir = fullfile(script_dir, '..', 'stim-videos');

for d = 1:4
    out_path = fullfile(stim_video_dir, names{d});
    vid = VideoWriter(out_path, 'MPEG-4');
    vid.FrameRate = opts.fps;
    vid.Quality = 95;
    open(vid);

    fprintf('Writing %s (%d frames) ...\n', labels{d}, num_frames);
    for f = 1:num_frames
        frame = repmat(gray, [S, S]);

        if d <= 2
            % --- Horizontal sweeps (LTR / RTL) ---
            if d == 1
                t = (f - 1) / max(1, num_frames - 1);
            else
                t = 1 - (f - 1) / max(1, num_frames - 1);
            end
            center_x = round(1 + t * (S - 1));

            x1 = max(1, center_x - floor(bw/2));
            x2 = min(S, center_x + floor(bw/2) - 1);
            bar_cols = x1:x2;
            checker_cols = bar_cols - (center_x - floor(bw/2)) + 1;
            checker_cols = max(1, min(bw, checker_cols));
            frame(:, bar_cols) = checker_v(:, checker_cols);
        else
            % --- Vertical sweeps (BTT / TTB) ---
            if d == 3
                t = 1 - (f - 1) / max(1, num_frames - 1);  % BTT: Y from S to 1
            else
                t = (f - 1) / max(1, num_frames - 1);        % TTB: Y from 1 to S
            end
            center_y = round(1 + t * (S - 1));

            y1 = max(1, center_y - floor(bw/2));
            y2 = min(S, center_y + floor(bw/2) - 1);
            bar_rows = y1:y2;
            checker_rows = bar_rows - (center_y - floor(bw/2)) + 1;
            checker_rows = max(1, min(bw, checker_rows));
            frame(bar_rows, :) = checker_h(checker_rows, :);
        end

        frame_rgb = cat(3, frame, frame, frame);
        writeVideo(vid, frame_rgb);
    end

    close(vid);
    fprintf('  Saved: %s\n', out_path);
end

fprintf('\nDone. 4 videos @ %dx%d, %d fps, %ds duration\n', S, S, opts.fps, opts.duration);

end
