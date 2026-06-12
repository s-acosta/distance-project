function stim_log = teensyStimulusSingleEye(opts)
% Present monocular stimulus to a SINGLE eye for all trials.
%
% Variant of teensyStimulusMonocular that stimulates only one eye (chosen
% via the 'eye' option) on every trial. Only that eye's video is generated
% and only that eye's Teensy port is opened; the other eye is not driven.
%
% Usage:
%   stim_log = teensyStimulusSingleEye()                    % default: right eye
%   stim_log = teensyStimulusSingleEye('eye', 'left')
%   stim_log = teensyStimulusSingleEye('eye', 'right', 'n_repeats', 6)
%   stim_log = teensyStimulusSingleEye('eye', 'left', 'preview', true)
%
% Returns:
%   stim_log.eye            - Which eye was stimulated ('right' or 'left')
%   stim_log.trial_start(i) - Start time of trial i (seconds)
%   stim_log.pre_start(i)   - Start of PRE phase
%   stim_log.stim_start(i)  - Start of stimulus video
%   stim_log.stim_end(i)    - End of stimulus video
%   stim_log.post_end(i)    - End of POST phase
%   stim_log.frame_times{i} - Frame timestamps for trial i

arguments
    opts.eye         (1,1) string  = "right"   % Which eye to stimulate: "right" or "left"
    opts.right_start (1,1) double  = 0          % Right eye start position (0=front)
    opts.right_end   (1,1) double  = 120        % Right eye end position
    opts.left_start  (1,1) double  = 160        % Left eye start position
    opts.left_end    (1,1) double  = 40          % Left eye end position (0=front)
    opts.stim_size   (1,1) double  = 30         % Stimulus circle diameter (pixels)
    opts.PORT        (1,1) string  = "COM6"     % Right eye port
    opts.PORT_left   (1,1) string  = "COM5"     % Left eye port
    opts.baud        (1,1) double  = 115200
    opts.H           (1,1) double  = 240
    opts.W           (1,1) double  = 240
    opts.n_repeats   (1,1) double  = 4          % Total trials (single eye)
    opts.pre_time    (1,1) double  = 2.0
    opts.post_time   (1,1) double  = 3.0
    opts.target_fps  (1,1) double  = 10
    opts.brightness  (1,1) double  = 1.0
    opts.DAQ_flag    (1,1) logical = false
    opts.scope       (1,1) string  = '2p'
    opts.preview     (1,1) logical = false
    opts.verbose     (1,1) logical = true
    opts.duration    (1,1) double  = 6.0        % Video duration (s)
    opts.frame_rate  (1,1) double  = 10         % Video generation frame rate
    opts.rcycles     (1,1) double  = 1
    opts.tcycles     (1,1) double  = 2
    opts.window_size (1,1) double  = 240
end

eye = lower(opts.eye);
if ~ismember(eye, ["right" "left"])
    error('opts.eye must be "right" or "left" (got "%s").', opts.eye);
end
is_right = strcmp(eye, "right");
eye_code = 1; if ~is_right, eye_code = 2; end   % 1=right, 2=left (matches teensyStimulusMonocular)

% Resolve per-eye start/end position and port
if is_right
    x_start = opts.right_start;  x_end = opts.right_end;  PORT = opts.PORT;
else
    x_start = opts.left_start;   x_end = opts.left_end;   PORT = opts.PORT_left;
end

total_trials   = opts.n_repeats;
frame_interval = 1 / opts.target_fps;

% === GENERATE VIDEO ===
func_dir  = fileparts(mfilename('fullpath'));
stim_name = fullfile(func_dir, sprintf('stim_%s_x%d_to_%d_size%d.mp4', ...
    eye, round(x_start), round(x_end), round(opts.stim_size)));

if opts.verbose
    fprintf('=== Generating stimulus video ===\n');
end

if ~isfile(stim_name)
    if opts.verbose
        fprintf('  %s eye: x=%d -> %d\n', eye, round(x_start), round(x_end));
    end
    generate_leftright_stim(x_start, x_end, opts.stim_size, ...
        opts.duration, opts.frame_rate, ...
        'output_name',  stim_name, ...
        'window_size',  opts.window_size, ...
        'rcycles',      opts.rcycles, ...
        'tcycles',      opts.tcycles);
else
    if opts.verbose
        fprintf('  Video already exists, skipping: %s\n', stim_name);
    end
end

% === PRE-COMPUTE FRAMES ===
if opts.verbose
    fprintf('\n=== Pre-computing frames ===\n');
end

v = VideoReader(stim_name);
vid_duration = v.Duration;

raw = {}; v.CurrentTime = 0;
while hasFrame(v)
    rgb = readFrame(v);
    if size(rgb,1) ~= opts.H || size(rgb,2) ~= opts.W
        rgb = imresize(rgb, [opts.H opts.W], 'nearest');
    end
    raw{end+1} = rgb; %#ok<AGROW>
end

frames   = process_frames(raw, char(eye), opts.brightness);
payloads = cellfun(@rgb565_payload, frames, 'UniformOutput', false);
n_frames = length(frames);

if opts.verbose
    fprintf('  %s eye: %d frames\n', eye, n_frames);
end

gray_img     = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), char(eye));
gray_payload = rgb565_payload(gray_img);

trial_dur = opts.pre_time + vid_duration + opts.post_time;
total_dur = trial_dur * total_trials;

if opts.verbose
    fprintf('\n=== Teensy Single-Eye Stimulus ===\n');
    fprintf('Eye: %s | x=%d->%d\n', eye, round(x_start), round(x_end));
    fprintf('Stim size: %d px | Duration: %.1f s @ %d fps\n', ...
        round(opts.stim_size), opts.duration, round(opts.frame_rate));
    fprintf('Trial structure: PRE %.1fs | STIM %.2fs | POST %.1fs (%.2fs total)\n', ...
        opts.pre_time, vid_duration, opts.post_time, trial_dur);
    fprintf('Total trials: %d\n', total_trials);
    fprintf('Total experiment: %.1f s (%.1f min)\n', total_dur, total_dur/60);
    fprintf('Brightness: %.2f\n', opts.brightness);
    fprintf('DAQ triggering: %s\n', string(opts.DAQ_flag));
    if opts.DAQ_flag, fprintf('Scope: %s\n', opts.scope); end
    if opts.preview
        fprintf('MODE: PREVIEW (screen only, no Teensy)\n');
    else
        fprintf('Port: %s\n', PORT);
    end
    fprintf('=================================\n\n');
end

% === SETUP ===
if opts.preview
    figure('Name', 'Single-Eye Preview', 'NumberTitle', 'off');
    ax = axes; imshow(gray_img, 'Parent', ax); title(ax, sprintf('%s eye', eye));
    ih = findobj(ax, 'Type', 'image');
    drawnow;
else
    if opts.DAQ_flag
        if strcmp(opts.scope, 'widefield')
            wf = WidefieldTriggerer(opts.DAQ_flag);
            wf.initialize(total_dur);
        elseif strcmp(opts.scope, '2p')
            DAQ = daq('ni');
            DAQ.addoutput('Dev1', 'ao0', 'Voltage');
            DAQ.write(0);
            fprintf('2P DAQ initialized\n');
        end
    end

    s_stim = serialport(PORT, opts.baud, "Timeout", 2);
    configureTerminator(s_stim, "LF"); flush(s_stim);
    s_stim.OutputBufferSize = 300000;
end

% === INITIALIZE LOG ===
stim_log = struct();
stim_log.parameters    = opts;
stim_log.eye           = char(eye);
stim_log.eye_order     = eye_code * ones(1, total_trials);   % 1=right, 2=left (all trials same eye)
stim_log.stim_duration = vid_duration;
stim_log.trial_start = zeros(1, total_trials);
stim_log.pre_start   = zeros(1, total_trials);
stim_log.stim_start  = zeros(1, total_trials);
stim_log.stim_end    = zeros(1, total_trials);
stim_log.post_end    = zeros(1, total_trials);
stim_log.frame_times = cell(1, total_trials);

if opts.verbose && ~opts.preview
    fprintf('Press Enter when ready...\n');
    pause();
end

% Start timing and trigger DAQ
if opts.DAQ_flag && ~opts.preview
    if strcmp(opts.scope, 'widefield')
        wf.start();
        experiment_start = tic;
    elseif strcmp(opts.scope, '2p')
        DAQ.write(5);
        experiment_start = tic;
        fprintf('2P DAQ trigger sent (5V)\n\n');
    end
else
    experiment_start = tic;
end

if opts.verbose
    fprintf('Starting experiment...\n\n');
end

% === MAIN TRIAL LOOP ===
for trial = 1:total_trials
    if opts.verbose
        fprintf('--- Trial %d/%d [eye: %s] ---\n', trial, total_trials, eye);
    end

    trial_frame_times = [];
    trial_offset = (trial - 1) * trial_dur;

    stim_log.trial_start(trial) = toc(experiment_start);

    % === PRE PHASE (gray) ===
    stim_log.pre_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  PRE start:  %.4f s\n', stim_log.pre_start(trial));
    end

    if opts.preview
        set(ih, 'CData', gray_img); title(ax, sprintf('Trial %d PRE', trial)); drawnow;
    end

    frame_count = 0;
    phase_end   = trial_offset + opts.pre_time;
    while toc(experiment_start) < phase_end
        if ~opts.preview
            writeline(s_stim, sprintf('IMG %d %d', opts.W, opts.H));
            write(s_stim, gray_payload, 'uint8');
        end
        frame_count = frame_count + 1;
        trial_frame_times(end+1) = toc(experiment_start); %#ok<AGROW>
        next_frame = trial_offset + frame_count * frame_interval;
        while toc(experiment_start) < next_frame, pause(0.0001); end
    end

    % === STIM PHASE ===
    stim_log.stim_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  STIM start: %.4f s\n', stim_log.stim_start(trial));
    end

    stim_frame_count = 0;
    stim_offset = trial_offset + opts.pre_time;
    phase_end   = stim_offset + vid_duration;

    if opts.preview
        title(ax, sprintf('Trial %d STIM', trial));
    end

    while toc(experiment_start) < phase_end
        frame_idx = mod(stim_frame_count, n_frames) + 1;

        if opts.preview
            set(ih, 'CData', frames{frame_idx}); drawnow;
        else
            writeline(s_stim, sprintf('IMG %d %d', opts.W, opts.H));
            write(s_stim, payloads{frame_idx}, 'uint8');
        end

        stim_frame_count = stim_frame_count + 1;
        trial_frame_times(end+1) = toc(experiment_start); %#ok<AGROW>
        target_time = stim_offset + stim_frame_count * frame_interval;
        while toc(experiment_start) < target_time, pause(0.0001); end
    end

    stim_log.stim_end(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  STIM end:   %.4f s (duration: %.4f s)\n', ...
            stim_log.stim_end(trial), stim_log.stim_end(trial) - stim_log.stim_start(trial));
    end

    % === POST PHASE (gray) ===
    post_offset      = trial_offset + opts.pre_time + vid_duration;
    phase_end        = trial_offset + trial_dur;
    post_frame_count = 0;

    if opts.preview
        set(ih, 'CData', gray_img); title(ax, sprintf('Trial %d POST', trial)); drawnow;
    end

    while toc(experiment_start) < phase_end
        if ~opts.preview
            writeline(s_stim, sprintf('IMG %d %d', opts.W, opts.H));
            write(s_stim, gray_payload, 'uint8');
        end
        post_frame_count = post_frame_count + 1;
        trial_frame_times(end+1) = toc(experiment_start); %#ok<AGROW>
        next_frame = post_offset + post_frame_count * frame_interval;
        while toc(experiment_start) < next_frame, pause(0.0001); end
    end

    stim_log.post_end(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  POST end:   %.4f s\n', stim_log.post_end(trial));
    end

    stim_log.frame_times{trial} = trial_frame_times;

    if opts.verbose
        trial_dur_actual = stim_log.post_end(trial) - stim_log.trial_start(trial);
        fprintf('  Trial complete: %d frames, %.2f s (%.2f fps)\n\n', ...
            length(trial_frame_times), trial_dur_actual, length(trial_frame_times)/trial_dur_actual);
    end
end

stim_log.finish_time = toc(experiment_start);

% === CLEANUP ===
if opts.preview
    set(ih, 'CData', gray_img); title(ax, 'Done'); drawnow;
else
    if opts.DAQ_flag
        if strcmp(opts.scope, 'widefield')
            wf.finish();
        elseif strcmp(opts.scope, '2p')
            DAQ.write(0);
            fprintf('2P DAQ trigger ended (0V)\n');
        end
    end
    clear s_stim;
end

total_frames = sum(cellfun(@length, stim_log.frame_times));
overall_fps  = total_frames / stim_log.finish_time;

if opts.verbose
    fprintf('\n=== Experiment Complete ===\n');
    fprintf('Total time: %.3f s (%.2f min)\n', stim_log.finish_time, stim_log.finish_time/60);
    fprintf('Total frames: %d | Overall FPS: %.2f\n', total_frames, overall_fps);
end

end

%% Helper functions
function out = process_frames(raw, eye_str, brightness)
    out = cell(1, length(raw));
    for i = 1:length(raw)
        f = uint8(round(double(raw{i}) * brightness));
        out{i} = rotate_for_eye(f, eye_str);
    end
end

function frame = rotate_for_eye(frame, eye)
    if strcmp(eye, 'right')
        frame = flip(frame, 2);
        frame = rot90(frame, -1);
    elseif strcmp(eye, 'left')
        frame = rot90(frame, 1);
    end
end

function payload = rgb565_payload(img)
    if ~isa(img, 'uint8'), img = im2uint8(img); end
    R5 = bitshift(uint16(img(:,:,1)), -3);
    G6 = bitshift(uint16(img(:,:,2)), -2);
    B5 = bitshift(uint16(img(:,:,3)), -3);
    rgb565 = bitor(bitshift(R5,11), bitor(bitshift(G6,5), B5));
    payload = typecast(reshape(rgb565.', [], 1), 'uint8');
end
