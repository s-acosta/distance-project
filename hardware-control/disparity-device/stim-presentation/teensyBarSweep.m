function stim_log = teensyBarSweep(ltr_path, rtl_path, opts)
% Play alternating left-to-right and right-to-left bar sweeps on Teensy.
%
% Mimics the signMapStim trial structure: for each repeat, plays both
% directions with gray blanks in between.
%
% Usage:
%   stim_log = teensyBarSweep('bar_left_to_right.mp4', 'bar_right_to_left.mp4')
%   stim_log = teensyBarSweep('bar_left_to_right.mp4', 'bar_right_to_left.mp4', 'n_repeats', 20)
%
% Returns:
%   stim_log.blank_on(rep, dir)   - Blank start time
%   stim_log.blank_end(rep, dir)  - Blank end time
%   stim_log.mov_on(rep, dir)     - Sweep start time
%   stim_log.mov_end(rep, dir)    - Sweep end time
%   stim_log.frame_times{rep, dir} - Frame timestamps per sweep

arguments
    ltr_path (1,1) string
    rtl_path (1,1) string
    opts.PORT (1,1) string = "COM3"
    opts.baud (1,1) double = 115200
    opts.H (1,1) double = 240
    opts.W (1,1) double = 240
    opts.n_repeats (1,1) double = 20
    opts.off_time (1,1) double = 2.0       % Gray screen between sweeps (seconds)
    opts.target_fps (1,1) double = 10
    opts.resize (1,1) logical = true
    opts.verbose (1,1) logical = true
    opts.DAQ_flag (1,1) logical = false
end

% Validate and open videos
assert(isfile(ltr_path), "Video not found: %s", ltr_path);
assert(isfile(rtl_path), "Video not found: %s", rtl_path);
videos = {VideoReader(ltr_path), VideoReader(rtl_path)};
dir_labels = {'Left-to-Right', 'Right-to-Left'};

frame_interval = 1 / opts.target_fps;

% Create gray frame
gray_frame = uint8(ones(opts.H, opts.W, 3) * 127);

% Calculate timing
on_time = videos{1}.Duration;
stim_dur = on_time + opts.off_time;
rep_dur = stim_dur * 2;
total_dur = rep_dur * opts.n_repeats;

if opts.verbose
    fprintf('=== Teensy Bar Sweep Stimulus ===\n');
    fprintf('Videos: %s, %s\n', ltr_path, rtl_path);
    fprintf('Sweep duration: %.2f s\n', on_time);
    fprintf('Blank duration: %.2f s\n', opts.off_time);
    fprintf('Repeats: %d\n', opts.n_repeats);
    fprintf('Total duration: %.2f s (%.2f min)\n', total_dur, total_dur/60);
    fprintf('Target FPS: %.2f\n', opts.target_fps);
    fprintf('DAQ triggering: %s\n', string(opts.DAQ_flag));
    fprintf('=================================\n\n');
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

% Initialize logging (matching signMapStim format)
stim_log = struct();
stim_log.parameters = opts;
stim_log.blank_on = zeros(opts.n_repeats, 2);
stim_log.blank_end = zeros(opts.n_repeats, 2);
stim_log.mov_on = zeros(opts.n_repeats, 2);
stim_log.mov_end = zeros(opts.n_repeats, 2);
stim_log.off_time = opts.off_time;
stim_log.on_time = on_time;
stim_log.repeats = opts.n_repeats;
stim_log.frame_times = cell(opts.n_repeats, 2);

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

% Main loop: repeats x directions
for rep = 1:opts.n_repeats
    for dir = 1:2
        if opts.verbose
            fprintf('--- Rep %d/%d, %s ---\n', rep, opts.n_repeats, dir_labels{dir});
        end

        trial_frame_times = [];

        % === BLANK PHASE (Gray screen) ===
        stim_log.blank_on(rep, dir) = toc(experiment_start);
        if opts.verbose
            fprintf('  Blank start: %.4f s\n', stim_log.blank_on(rep, dir));
        end

        blank_end_target = stim_log.blank_on(rep, dir) + opts.off_time;
        while toc(experiment_start) < blank_end_target
            payload = rgb565_payload(gray_frame);
            writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
            write(s, payload, 'uint8');

            trial_frame_times(end+1) = toc(experiment_start);

            next_frame = stim_log.blank_on(rep, dir) + length(trial_frame_times) * frame_interval;
            while toc(experiment_start) < next_frame
                pause(0.0001);
            end
        end

        stim_log.blank_end(rep, dir) = toc(experiment_start);

        % === SWEEP PHASE (Video) ===
        stim_log.mov_on(rep, dir) = toc(experiment_start);
        if opts.verbose
            fprintf('  Sweep start: %.4f s\n', stim_log.mov_on(rep, dir));
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

            trial_frame_times(end+1) = toc(experiment_start);
            stim_frame_count = stim_frame_count + 1;

            target_time = stim_log.mov_on(rep, dir) + stim_frame_count * frame_interval;
            while toc(experiment_start) < target_time
                pause(0.0001);
            end
        end

        stim_log.mov_end(rep, dir) = toc(experiment_start);
        stim_log.frame_times{rep, dir} = trial_frame_times;

        if opts.verbose
            fprintf('  Sweep end:   %.4f s (duration: %.4f s)\n', ...
                stim_log.mov_end(rep, dir), stim_log.mov_end(rep, dir) - stim_log.mov_on(rep, dir));
        end
    end

    if opts.verbose
        fprintf('\n');
    end
end

stim_log.finish_time = toc(experiment_start);

% Turn off DAQ trigger
if opts.DAQ_flag
    DAQ.write(0);
    fprintf('DAQ trigger ended (0V)\n');
end

if opts.verbose
    total_frames = sum(cellfun(@length, stim_log.frame_times));
    fprintf('\n=== Experiment Complete ===\n');
    fprintf('Total time: %.3f s (%.2f min)\n', stim_log.finish_time, stim_log.finish_time/60);
    fprintf('Total frames: %d\n', total_frames);
    fprintf('Overall FPS: %.2f\n', total_frames / stim_log.finish_time);
end

clear s;
for v_idx = 1:2, delete(videos{v_idx}); end

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
