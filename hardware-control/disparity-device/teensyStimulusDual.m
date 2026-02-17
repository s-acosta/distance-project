function stim_log = teensyStimulusDual(video_path_left, video_path_right, opts)
% Play different videos on two Teensy-driven screens (left/right eye)
%
% Usage:
%   stim_log = teensyStimulusDual('left.mp4', 'right.mp4')
%   stim_log = teensyStimulusDual('left.mp4', 'right.mp4', ...
%       'PORT_LEFT', 'COM3', 'PORT_RIGHT', 'COM4', 'n_repeats', 5)
%
% Returns:
%   stim_log.trial_start(i)   - Start time of trial i (seconds)
%   stim_log.pre_start(i)     - Start of PRE phase (gray screen)
%   stim_log.stim_start(i)    - Start of stimulus video
%   stim_log.stim_end(i)      - End of stimulus video
%   stim_log.post_end(i)      - End of POST phase (gray screen)
%   stim_log.frame_times{i}   - Frame timestamps for trial i

arguments
    video_path_left (1,1) string
    video_path_right (1,1) string
    opts.PORT_LEFT (1,1) string = "COM3"
    opts.PORT_RIGHT (1,1) string = "COM4"
    opts.baud (1,1) double = 115200
    opts.H (1,1) double = 240
    opts.W (1,1) double = 240
    opts.n_repeats (1,1) double = 2
    opts.pre_time (1,1) double = 2.0
    opts.post_time (1,1) double = 2.0
    opts.target_fps (1,1) double = 10
    opts.resize (1,1) logical = true
    opts.verbose (1,1) logical = true
    opts.DAQ_flag (1,1) logical = false
end

% Validate and open videos
assert(isfile(video_path_left), "Left video not found: %s", video_path_left);
assert(isfile(video_path_right), "Right video not found: %s", video_path_right);
v_left = VideoReader(video_path_left);
v_right = VideoReader(video_path_right);

% Determine target fps
if isempty(opts.target_fps)
    opts.target_fps = v_left.FrameRate;
end
frame_interval = 1 / opts.target_fps;

% Create gray frame
gray_frame = uint8(ones(opts.H, opts.W, 3) * 127);
gray_payload = rgb565_payload(gray_frame);

if opts.verbose
    fprintf('=== Teensy Dual-Screen Stimulus ===\n');
    fprintf('Left video:  %s | %dx%d @ %.2f fps (%.2f s)\n', ...
        video_path_left, v_left.Height, v_left.Width, v_left.FrameRate, v_left.Duration);
    fprintf('Right video: %s | %dx%d @ %.2f fps (%.2f s)\n', ...
        video_path_right, v_right.Height, v_right.Width, v_right.FrameRate, v_right.Duration);
    fprintf('Target FPS: %.2f (%.2f ms/frame)\n', opts.target_fps, 1000*frame_interval);
    fprintf('Trial structure:\n');
    fprintf('  PRE:  %.2f s (gray)\n', opts.pre_time);
    fprintf('  STIM: %.2f s (video, using shorter of the two)\n', min(v_left.Duration, v_right.Duration));
    fprintf('  POST: %.2f s (gray)\n', opts.post_time);
    fprintf('Number of repeats: %d\n', opts.n_repeats);
    fprintf('Ports: LEFT=%s  RIGHT=%s\n', opts.PORT_LEFT, opts.PORT_RIGHT);
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

% Open serial ports
s_left = serialport(opts.PORT_LEFT, opts.baud, "Timeout", 2);
configureTerminator(s_left, "LF");
flush(s_left);
s_left.OutputBufferSize = 300000;

s_right = serialport(opts.PORT_RIGHT, opts.baud, "Timeout", 2);
configureTerminator(s_right, "LF");
flush(s_right);
s_right.OutputBufferSize = 300000;

% Initialize logging
stim_log = struct();
stim_log.parameters = opts;
stim_log.video_left = video_path_left;
stim_log.video_right = video_path_right;
stim_log.trial_start = zeros(1, opts.n_repeats);
stim_log.pre_start = zeros(1, opts.n_repeats);
stim_log.stim_start = zeros(1, opts.n_repeats);
stim_log.stim_end = zeros(1, opts.n_repeats);
stim_log.post_end = zeros(1, opts.n_repeats);
stim_log.frame_times = cell(1, opts.n_repeats);

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

% Main trial loop
for trial = 1:opts.n_repeats
    if opts.verbose
        fprintf('--- Trial %d/%d ---\n', trial, opts.n_repeats);
    end

    trial_frame_times = [];

    % Mark trial start
    stim_log.trial_start(trial) = toc(experiment_start);

    % === PRE PHASE (Gray screen) ===
    stim_log.pre_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  PRE start:  %.4f s\n', stim_log.pre_start(trial));
    end

    phase_end = stim_log.pre_start(trial) + opts.pre_time;
    while toc(experiment_start) < phase_end
        send_frame(s_left, opts.W, opts.H, gray_payload);
        send_frame(s_right, opts.W, opts.H, gray_payload);

        trial_frame_times(end+1) = toc(experiment_start);

        % Maintain frame rate
        next_frame = stim_log.pre_start(trial) + length(trial_frame_times) * frame_interval;
        while toc(experiment_start) < next_frame
            pause(0.0001);
        end
    end

    % === STIMULUS PHASE (Video) ===
    stim_log.stim_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  STIM start: %.4f s\n', stim_log.stim_start(trial));
    end

    % Reset videos to beginning
    v_left.CurrentTime = 0;
    v_right.CurrentTime = 0;

    while hasFrame(v_left) && hasFrame(v_right)
        rgb_left = readFrame(v_left);
        rgb_right = readFrame(v_right);

        if opts.resize
            if size(rgb_left,1) ~= opts.H || size(rgb_left,2) ~= opts.W
                rgb_left = imresize(rgb_left, [opts.H opts.W], "nearest");
            end
            if size(rgb_right,1) ~= opts.H || size(rgb_right,2) ~= opts.W
                rgb_right = imresize(rgb_right, [opts.H opts.W], "nearest");
            end
        end

        payload_left = rgb565_payload(rgb_left);
        payload_right = rgb565_payload(rgb_right);
        send_frame(s_left, opts.W, opts.H, payload_left);
        send_frame(s_right, opts.W, opts.H, payload_right);

        trial_frame_times(end+1) = toc(experiment_start);

        % Maintain frame rate
        target_time = stim_log.stim_start(trial) + ...
            (length(trial_frame_times) - sum(trial_frame_times < stim_log.stim_start(trial))) * frame_interval;
        while toc(experiment_start) < target_time
            pause(0.0001);
        end
    end

    stim_log.stim_end(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  STIM end:   %.4f s (duration: %.4f s)\n', ...
            stim_log.stim_end(trial), stim_log.stim_end(trial) - stim_log.stim_start(trial));
    end

    % === POST PHASE (Gray screen) ===
    post_start = toc(experiment_start);
    phase_end = post_start + opts.post_time;

    while toc(experiment_start) < phase_end
        send_frame(s_left, opts.W, opts.H, gray_payload);
        send_frame(s_right, opts.W, opts.H, gray_payload);

        trial_frame_times(end+1) = toc(experiment_start);

        % Maintain frame rate
        next_frame = post_start + sum(trial_frame_times >= post_start) * frame_interval;
        while toc(experiment_start) < next_frame
            pause(0.0001);
        end
    end

    stim_log.post_end(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  POST end:   %.4f s\n', stim_log.post_end(trial));
    end

    % Store frame times for this trial
    stim_log.frame_times{trial} = trial_frame_times;

    if opts.verbose
        trial_duration = stim_log.post_end(trial) - stim_log.trial_start(trial);
        fprintf('  Trial complete: %d frames, %.2f s (%.2f fps)\n\n', ...
            length(trial_frame_times), trial_duration, length(trial_frame_times)/trial_duration);
    end
end

stim_log.finish_time = toc(experiment_start);

% Turn off DAQ trigger
if opts.DAQ_flag
    DAQ.write(0);
    fprintf('DAQ trigger ended (0V)\n');
end

% Calculate statistics
total_frames = sum(cellfun(@length, stim_log.frame_times));
overall_fps = total_frames / stim_log.finish_time;

if opts.verbose
    fprintf('\n=== Experiment Complete ===\n');
    fprintf('Total time: %.3f s (%.2f min)\n', stim_log.finish_time, stim_log.finish_time/60);
    fprintf('Total frames: %d\n', total_frames);
    fprintf('Overall FPS: %.2f\n', overall_fps);
end

clear s_left s_right v_left v_right;
end

%% Helper functions
function send_frame(s, W, H, payload)
    writeline(s, sprintf('IMG %d %d', W, H));
    write(s, payload, 'uint8');
end

function payload = rgb565_payload(img)
    if ~isa(img,'uint8'), img = im2uint8(img); end
    R5 = bitshift(uint16(img(:,:,1)), -3);
    G6 = bitshift(uint16(img(:,:,2)), -2);
    B5 = bitshift(uint16(img(:,:,3)), -3);
    rgb565 = bitor(bitshift(R5,11), bitor(bitshift(G6,5), B5));
    payload = typecast(reshape(rgb565.', [], 1), 'uint8');
end
