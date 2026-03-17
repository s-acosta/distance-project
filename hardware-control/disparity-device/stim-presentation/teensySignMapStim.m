function Stimdata = teensySignMapStim(opts)
% Play 4-direction bar sweeps on Teensy for sign mapping.
%
% Automatically finds bar sweep videos in the same directory as this script
% (bar_left_to_right.mp4, bar_right_to_left.mp4, bar_bottom_to_top.mp4,
% bar_top_to_bottom.mp4). Generate them first with generate_4direction_bar().
%
% Mimics the signMapStim trial structure exactly: for each repeat, plays
% all 4 directions with gray blanks in between. Outputs a Stimdata struct
% compatible with SignMapper_disk analysis pipeline.
%
% Direction order (matching SignMapper_disk column indices):
%   1 = forward  (left-to-right)
%   2 = backward (right-to-left)
%   3 = up       (bottom-to-top)
%   4 = down     (top-to-bottom)
%
% Usage:
%   teensySignMapStim()
%   teensySignMapStim('n_repeats', 20, 'off_time', 2, 'DAQ_flag', true)
%
% Saves Stimdata.mat via uisave with fields:
%   Stimdata.blank_on   (n_repeats x 4)  - Blank start times (seconds)
%   Stimdata.blank_end  (n_repeats x 4)  - Blank end times (seconds)
%   Stimdata.mov_on     (n_repeats x 4)  - Sweep start times (seconds)
%   Stimdata.mov_end    (n_repeats x 4)  - Sweep end times (seconds)
%   Stimdata.off_time   scalar            - Blank duration (seconds)
%   Stimdata.repeats    scalar            - Number of repeats
%   Stimdata.on_time    scalar            - Sweep duration (seconds)

arguments
    opts.PORT (1,1) string = "COM5"
    opts.baud (1,1) double = 115200
    opts.H (1,1) double = 240
    opts.W (1,1) double = 240
    opts.n_repeats (1,1) double = 20
    opts.off_time (1,1) double = 2.0
    opts.target_fps (1,1) double = 10
    opts.resize (1,1) logical = true
    opts.verbose (1,1) logical = true
    opts.DAQ_flag (1,1) logical = false
end

% Find videos in the stim-videos directory
script_dir = fileparts(mfilename('fullpath'));
stim_video_dir = fullfile(script_dir, '..', 'stim-videos');
video_names = {'bar_left_to_right.mp4', 'bar_right_to_left.mp4', ...
               'bar_bottom_to_top.mp4', 'bar_top_to_bottom.mp4'};
paths = cellfun(@(n) fullfile(stim_video_dir, n), video_names, 'UniformOutput', false);
for i = 1:4
    assert(isfile(paths{i}), "Video not found: %s\nRun generate_4direction_bar() first.", paths{i});
end
videos = cellfun(@VideoReader, paths, 'UniformOutput', false);
dir_labels = {'Forward (LTR)', 'Backward (RTL)', 'Upward (BTT)', 'Downward (TTB)'};

% Verify all videos have the same duration
durations = cellfun(@(v) v.Duration, videos);
assert(all(abs(durations - durations(1)) < 0.01), ...
    'All 4 videos must have the same duration (got: %s)', mat2str(durations, 3));

frame_interval = 1 / opts.target_fps;

% Create gray frame
gray_frame = uint8(ones(opts.H, opts.W, 3) * 127);

% Calculate timing
on_time = videos{1}.Duration;
stim_dur = on_time + opts.off_time;
rep_dur = stim_dur * 4;
total_dur = rep_dur * opts.n_repeats;

if opts.verbose
    fprintf('=== Teensy Sign Mapping Stimulus ===\n');
    fprintf('Videos:\n');
    for i = 1:4
        fprintf('  %s: %s\n', dir_labels{i}, paths{i});
    end
    fprintf('Sweep duration: %.2f s\n', on_time);
    fprintf('Blank duration: %.2f s\n', opts.off_time);
    fprintf('Repeats: %d\n', opts.n_repeats);
    fprintf('Total duration: %.2f s (%.2f min)\n', total_dur, total_dur/60);
    fprintf('Target FPS: %.2f\n', opts.target_fps);
    fprintf('DAQ triggering: %s\n', string(opts.DAQ_flag));
    fprintf('====================================\n\n');
end

% Initialize DAQ if requested
if opts.DAQ_flag
    DAQ = daq('ni');
    DAQ.addoutput('Dev1', 'ao0', 'Voltage');
    DAQ.write(0);
    fprintf('DAQ initialized\n');
end

% Open serial
s = serialport(opts.PORT, opts.baud, "Timeout", 2);
configureTerminator(s, "LF");
flush(s);
s.OutputBufferSize = 300000;

% Initialize logging arrays (matching signMapStim format: n_repeats x 4)
blank_on  = zeros(opts.n_repeats, 4);
blank_end = zeros(opts.n_repeats, 4);
mov_on    = zeros(opts.n_repeats, 4);
mov_end   = zeros(opts.n_repeats, 4);
frame_times = cell(opts.n_repeats, 4);

if opts.verbose
    fprintf('Press Enter when ready...\n');
    pause();
end

% Start timing and trigger DAQ
experiment_start = tic;

if opts.DAQ_flag
    DAQ.write(5);
    fprintf('DAQ trigger sent (5V)\n\n');
end

if opts.verbose
    fprintf('Starting experiment...\n\n');
end

% Main loop: repeats x 4 directions
for rep = 1:opts.n_repeats
    for dir = 1:4
        if opts.verbose
            fprintf('--- Rep %d/%d, %s ---\n', rep, opts.n_repeats, dir_labels{dir});
        end

        trial_frame_times = [];

        % === BLANK PHASE (Gray screen) ===
        blank_on(rep, dir) = toc(experiment_start);
        if opts.verbose
            fprintf('  Blank start: %.4f s\n', blank_on(rep, dir));
        end

        blank_end_target = blank_on(rep, dir) + opts.off_time;
        while toc(experiment_start) < blank_end_target
            payload = rgb565_payload(gray_frame);
            writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
            write(s, payload, 'uint8');

            trial_frame_times(end+1) = toc(experiment_start); %#ok<AGROW>

            next_frame = blank_on(rep, dir) + length(trial_frame_times) * frame_interval;
            while toc(experiment_start) < next_frame
                pause(0.0001);
            end
        end

        blank_end(rep, dir) = toc(experiment_start);

        % === SWEEP PHASE (Video) ===
        mov_on(rep, dir) = toc(experiment_start);
        if opts.verbose
            fprintf('  Sweep start: %.4f s\n', mov_on(rep, dir));
        end

        videos{dir}.CurrentTime = 0;
        stim_frame_count = 0;

        while hasFrame(videos{dir})
            rgb = readFrame(videos{dir});

            if opts.resize && (size(rgb,1) ~= opts.H || size(rgb,2) ~= opts.W)
                rgb = imresize(rgb, [opts.H opts.W], "nearest");
            end

            payload = rgb565_payload(rgb);
            writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
            write(s, payload, 'uint8');

            trial_frame_times(end+1) = toc(experiment_start); %#ok<AGROW>
            stim_frame_count = stim_frame_count + 1;

            target_time = mov_on(rep, dir) + stim_frame_count * frame_interval;
            while toc(experiment_start) < target_time
                pause(0.0001);
            end
        end

        mov_end(rep, dir) = toc(experiment_start);
        frame_times{rep, dir} = trial_frame_times;

        if opts.verbose
            fprintf('  Sweep end:   %.4f s (duration: %.4f s)\n', ...
                mov_end(rep, dir), mov_end(rep, dir) - mov_on(rep, dir));
        end
    end

    if opts.verbose
        fprintf('\n');
    end
end

finish_time = toc(experiment_start);

% Turn off DAQ trigger
if opts.DAQ_flag
    DAQ.write(0);
    fprintf('DAQ trigger ended (0V)\n');
end

if opts.verbose
    total_frames = sum(cellfun(@length, frame_times));
    fprintf('\n=== Experiment Complete ===\n');
    fprintf('Total time: %.3f s (%.2f min)\n', finish_time, finish_time/60);
    fprintf('Total frames: %d\n', total_frames);
    fprintf('Overall FPS: %.2f\n', total_frames / finish_time);
end

% Clean up serial and video readers
clear s;
for v_idx = 1:4, delete(videos{v_idx}); end

% Build Stimdata struct (matching signMapStim.m output exactly)
Stimdata.blank_on  = blank_on;
Stimdata.blank_end = blank_end;
Stimdata.mov_on    = mov_on;
Stimdata.mov_end   = mov_end;
Stimdata.off_time  = opts.off_time;
Stimdata.repeats   = opts.n_repeats;
Stimdata.on_time   = on_time;

% Save
uisave('Stimdata');

end

%% Helper function
function payload = rgb565_payload(img)
    if ~isa(img,'uint8'), img = im2uint8(img); end
    R5 = bitshift(uint16(img(:,:,1)), -3);
    G6 = bitshift(uint16(img(:,:,2)), -2);
    B5 = bitshift(uint16(img(:,:,3)), -3);
    rgb565 = bitor(bitshift(R5,11), bitor(bitshift(G6,5), B5));
    payload = typecast(reshape(rgb565.', [], 1), 'uint8');
end
