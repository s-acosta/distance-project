function stim_log = teensyStimulusMonocular(opts)
% Present monocular stimulus to each eye in separate trials.
%
% Generates one video per eye via generate_leftright_stim, then runs a
% trial sequence where each trial stimulates one eye while the other shows
% gray. Trial order uses alternating pairs ([R,L], [L,R], [R,L], ...) so
% each eye gets exactly n_repeats trials, never the same eye more than
% twice in a row.
%
% Usage:
%   stim_log = teensyStimulusMonocular()
%   stim_log = teensyStimulusMonocular('n_repeats', 4, 'stim_size', 20)
%   stim_log = teensyStimulusMonocular('preview', true)
%
% Returns:
%   stim_log.eye_order(i)     - Which eye was active on trial i (1=right, 2=left)
%   stim_log.trial_start(i)   - Start time of trial i (seconds)
%   stim_log.pre_start(i)     - Start of PRE phase
%   stim_log.stim_start(i)    - Start of stimulus video
%   stim_log.stim_end(i)      - End of stimulus video
%   stim_log.post_end(i)      - End of POST phase
%   stim_log.frame_times{i}   - Frame timestamps for trial i

arguments
    opts.right_start (1,1) double  = 0        % Right eye start position (0=front)
    opts.right_end   (1,1) double  = 120      % Right eye end position
    opts.left_start  (1,1) double  = 120      % Left eye start position
    opts.left_end    (1,1) double  = 0        % Left eye end position (0=front)
    opts.stim_size   (1,1) double  = 30       % Stimulus circle diameter (pixels)
    opts.PORT        (1,1) string  = "COM6"
    opts.PORT_left   (1,1) string  = "COM5"
    opts.baud        (1,1) double  = 115200
    opts.H           (1,1) double  = 240
    opts.W           (1,1) double  = 240
    opts.n_repeats   (1,1) double  = 20        % Trials per eye (total = 2 * n_repeats)
    opts.pre_time    (1,1) double  = 2.0
    opts.post_time   (1,1) double  = 3.0
    opts.target_fps  (1,1) double  = 10
    opts.brightness  (1,1) double  = 0.6
    opts.DAQ_flag    (1,1) logical = true
    opts.scope       (1,1) string  = '2p'
    opts.preview     (1,1) logical = false
    opts.verbose     (1,1) logical = true
    opts.duration    (1,1) double  = 6.0      % Video duration (s)
    opts.frame_rate  (1,1) double  = 10       % Video generation frame rate
    opts.rcycles     (1,1) double  = 1
    opts.tcycles     (1,1) double  = 2
    opts.window_size (1,1) double  = 240
end

total_trials   = 2 * opts.n_repeats;
frame_interval = 1 / opts.target_fps;

% === GENERATE VIDEOS ===
func_dir   = fileparts(mfilename('fullpath'));
right_name = fullfile(func_dir, sprintf('stim_right_x%d_to_%d_size%d.mp4', ...
    round(opts.right_start), round(opts.right_end), round(opts.stim_size)));
left_name  = fullfile(func_dir, sprintf('stim_left_x%d_to_%d_size%d.mp4', ...
    round(opts.left_start),  round(opts.left_end),  round(opts.stim_size)));

if opts.verbose
    fprintf('=== Generating stimulus videos ===\n');
end

if ~isfile(right_name)
    if opts.verbose
        fprintf('  Right eye: x=%d -> %d\n', round(opts.right_start), round(opts.right_end));
    end
    generate_leftright_stim(opts.right_start, opts.right_end, opts.stim_size, ...
        opts.duration, opts.frame_rate, ...
        'output_name',  right_name, ...
        'window_size',  opts.window_size, ...
        'rcycles',      opts.rcycles, ...
        'tcycles',      opts.tcycles);
else
    if opts.verbose
        fprintf('  Right video already exists, skipping: %s\n', right_name);
    end
end

if ~isfile(left_name)
    if opts.verbose
        fprintf('  Left eye:  x=%d -> %d\n', round(opts.left_start), round(opts.left_end));
    end
    generate_leftright_stim(opts.left_start, opts.left_end, opts.stim_size, ...
        opts.duration, opts.frame_rate, ...
        'output_name',  left_name, ...
        'window_size',  opts.window_size, ...
        'rcycles',      opts.rcycles, ...
        'tcycles',      opts.tcycles);
else
    if opts.verbose
        fprintf('  Left video already exists, skipping: %s\n', left_name);
    end
end

% === PRE-COMPUTE FRAMES ===
if opts.verbose
    fprintf('\n=== Pre-computing frames ===\n');
end

vR = VideoReader(right_name);
vL = VideoReader(left_name);
vid_duration = vR.Duration;

raw_R = {}; vR.CurrentTime = 0;
while hasFrame(vR)
    rgb = readFrame(vR);
    if size(rgb,1) ~= opts.H || size(rgb,2) ~= opts.W
        rgb = imresize(rgb, [opts.H opts.W], 'nearest');
    end
    raw_R{end+1} = rgb; %#ok<AGROW>
end

raw_L = {}; vL.CurrentTime = 0;
while hasFrame(vL)
    rgb = readFrame(vL);
    if size(rgb,1) ~= opts.H || size(rgb,2) ~= opts.W
        rgb = imresize(rgb, [opts.H opts.W], 'nearest');
    end
    raw_L{end+1} = rgb; %#ok<AGROW>
end

frames_R   = process_frames(raw_R, 'right', opts.brightness);
frames_L   = process_frames(raw_L, 'left',  opts.brightness);
payloads_R = cellfun(@rgb565_payload, frames_R, 'UniformOutput', false);
payloads_L = cellfun(@rgb565_payload, frames_L, 'UniformOutput', false);
n_frames_R = length(frames_R);
n_frames_L = length(frames_L);

if opts.verbose
    fprintf('  Right eye: %d frames | Left eye: %d frames\n', n_frames_R, n_frames_L);
end

gray_img_R     = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), 'right');
gray_img_L     = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), 'left');
gray_payload_R = rgb565_payload(gray_img_R);
gray_payload_L = rgb565_payload(gray_img_L);

% === EYE SEQUENCE: alternating pairs [1,2],[2,1],[1,2],... ===
eye_seq = zeros(1, total_trials);
for p = 1:opts.n_repeats
    idx = (p-1)*2 + 1;
    if mod(p, 2) == 1
        eye_seq(idx:idx+1) = [1, 2];
    else
        eye_seq(idx:idx+1) = [2, 1];
    end
end
% eye_seq: 1=right active, 2=left active

trial_dur = opts.pre_time + vid_duration + opts.post_time;
total_dur  = trial_dur * total_trials;

if opts.verbose
    fprintf('\n=== Teensy Monocular Stimulus ===\n');
    fprintf('Right eye: x=%d->%d | Left eye: x=%d->%d\n', ...
        round(opts.right_start), round(opts.right_end), round(opts.left_start), round(opts.left_end));
    fprintf('Stim size: %d px | Duration: %.1f s @ %d fps\n', ...
        round(opts.stim_size), opts.duration, round(opts.frame_rate));
    fprintf('Trial structure: PRE %.1fs | STIM %.2fs | POST %.1fs (%.2fs total)\n', ...
        opts.pre_time, vid_duration, opts.post_time, trial_dur);
    fprintf('n_repeats per eye: %d | Total trials: %d\n', opts.n_repeats, total_trials);
    fprintf('Total experiment: %.1f s (%.1f min)\n', total_dur, total_dur/60);
    fprintf('Eye sequence (1=R,2=L): %s\n', mat2str(eye_seq));
    fprintf('Brightness: %.2f\n', opts.brightness);
    fprintf('DAQ triggering: %s\n', string(opts.DAQ_flag));
    if opts.DAQ_flag, fprintf('Scope: %s\n', opts.scope); end
    if opts.preview
        fprintf('MODE: PREVIEW (screen only, no Teensy)\n');
    else
        fprintf('Right port: %s | Left port: %s\n', opts.PORT, opts.PORT_left);
    end
    fprintf('=================================\n\n');
end

% === SETUP ===
if opts.preview
    figure('Name', 'Monocular Preview', 'NumberTitle', 'off');
    ax_L = subplot(1,2,1); imshow(gray_img_L, 'Parent', ax_L); title(ax_L, 'Left eye');
    ax_R = subplot(1,2,2); imshow(gray_img_R, 'Parent', ax_R); title(ax_R, 'Right eye');
    ih_R = findobj(ax_R, 'Type', 'image');
    ih_L = findobj(ax_L, 'Type', 'image');
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

    s_right = serialport(opts.PORT,      opts.baud, "Timeout", 2);
    configureTerminator(s_right, "LF"); flush(s_right);
    s_right.OutputBufferSize = 300000;

    s_left  = serialport(opts.PORT_left, opts.baud, "Timeout", 2);
    configureTerminator(s_left,  "LF"); flush(s_left);
    s_left.OutputBufferSize = 300000;
end

% === INITIALIZE LOG ===
stim_log = struct();
stim_log.parameters  = opts;
stim_log.stim_duration = vid_duration;
stim_log.eye_order     = eye_seq;
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
    active_eye = eye_seq(trial);   % 1=right, 2=left

    if opts.verbose
        eye_label = 'right'; if active_eye == 2, eye_label = 'left'; end
        fprintf('--- Trial %d/%d [active eye: %s] ---\n', trial, total_trials, eye_label);
    end

    % Resolve assets for active/idle eye
    if active_eye == 1   % right eye active
        stim_payloads = payloads_R;
        stim_frames   = frames_R;
        n_frames      = n_frames_R;
        if ~opts.preview
            s_stim = s_right;  s_idle = s_left;
            gray_payload_idle = gray_payload_L;
        end
        active_gray_img = gray_img_R;
        idle_gray_img   = gray_img_L;
    else                 % left eye active
        stim_payloads = payloads_L;
        stim_frames   = frames_L;
        n_frames      = n_frames_L;
        if ~opts.preview
            s_stim = s_left;   s_idle = s_right;
            gray_payload_idle = gray_payload_R;
        end
        active_gray_img = gray_img_L;
        idle_gray_img   = gray_img_R;
    end

    trial_frame_times = [];
    trial_offset = (trial - 1) * trial_dur;

    stim_log.trial_start(trial) = toc(experiment_start);

    % === PRE PHASE (gray both eyes) ===
    stim_log.pre_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  PRE start:  %.4f s\n', stim_log.pre_start(trial));
    end

    if opts.preview
        if active_eye == 1
            set(ih_R, 'CData', gray_img_R); title(ax_R, sprintf('Trial %d PRE (active)', trial));
            set(ih_L, 'CData', gray_img_L); title(ax_L, sprintf('Trial %d PRE (idle)',   trial));
        else
            set(ih_L, 'CData', gray_img_L); title(ax_L, sprintf('Trial %d PRE (active)', trial));
            set(ih_R, 'CData', gray_img_R); title(ax_R, sprintf('Trial %d PRE (idle)',   trial));
        end
        drawnow;
    end

    frame_count = 0;
    phase_end   = trial_offset + opts.pre_time;
    while toc(experiment_start) < phase_end
        if ~opts.preview
            writeline(s_right, sprintf('IMG %d %d', opts.W, opts.H));
            write(s_right, gray_payload_R, 'uint8');
            writeline(s_left,  sprintf('IMG %d %d', opts.W, opts.H));
            write(s_left,  gray_payload_L, 'uint8');
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
        if active_eye == 1
            title(ax_R, sprintf('Trial %d STIM (active)', trial));
            title(ax_L, sprintf('Trial %d STIM (idle)',   trial));
        else
            title(ax_L, sprintf('Trial %d STIM (active)', trial));
            title(ax_R, sprintf('Trial %d STIM (idle)',   trial));
        end
    end

    while toc(experiment_start) < phase_end
        frame_idx = mod(stim_frame_count, n_frames) + 1;

        if opts.preview
            if active_eye == 1
                set(ih_R, 'CData', stim_frames{frame_idx});
                set(ih_L, 'CData', idle_gray_img);
            else
                set(ih_L, 'CData', stim_frames{frame_idx});
                set(ih_R, 'CData', idle_gray_img);
            end
            drawnow;
        else
            writeline(s_stim, sprintf('IMG %d %d', opts.W, opts.H));
            write(s_stim, stim_payloads{frame_idx}, 'uint8');
            writeline(s_idle, sprintf('IMG %d %d', opts.W, opts.H));
            write(s_idle, gray_payload_idle, 'uint8');
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

    % === POST PHASE (gray both eyes) ===
    post_offset      = trial_offset + opts.pre_time + vid_duration;
    phase_end        = trial_offset + trial_dur;
    post_frame_count = 0;

    if opts.preview
        if active_eye == 1
            set(ih_R, 'CData', gray_img_R); title(ax_R, sprintf('Trial %d POST (active)', trial));
            set(ih_L, 'CData', gray_img_L); title(ax_L, sprintf('Trial %d POST (idle)',   trial));
        else
            set(ih_L, 'CData', gray_img_L); title(ax_L, sprintf('Trial %d POST (active)', trial));
            set(ih_R, 'CData', gray_img_R); title(ax_R, sprintf('Trial %d POST (idle)',   trial));
        end
        drawnow;
    end

    while toc(experiment_start) < phase_end
        if ~opts.preview
            writeline(s_right, sprintf('IMG %d %d', opts.W, opts.H));
            write(s_right, gray_payload_R, 'uint8');
            writeline(s_left,  sprintf('IMG %d %d', opts.W, opts.H));
            write(s_left,  gray_payload_L, 'uint8');
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
    set(ih_R, 'CData', gray_img_R); title(ax_R, 'Done');
    set(ih_L, 'CData', gray_img_L); title(ax_L, 'Done');
    drawnow;
else
    if opts.DAQ_flag
        if strcmp(opts.scope, 'widefield')
            wf.finish();
        elseif strcmp(opts.scope, '2p')
            DAQ.write(0);
            fprintf('2P DAQ trigger ended (0V)\n');
        end
    end
    clear s_right s_left;
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
