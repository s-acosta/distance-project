function stim_log = teensyStimulusBoth(start_point, end_point, stim_size, opts)
% Present binocular stimulus on two Teensys simultaneously across disparity conditions.
%
% The right eye shows the stimulus at each disparity offset; the left eye always
% shows the opposite-direction motion (binocular coherence). Both screens receive
% frames every trial. Send order alternates per trial to counterbalance the
% ~30-60 ms serial desync inherent to sequential USB-CDC transfers.
%
% Usage:
%   stim_log = teensyStimulusBoth(0, 240, 20)
%   stim_log = teensyStimulusBoth(0, 240, 20, 'disparities', [-20 0 20], 'n_repeats', 4)
%   stim_log = teensyStimulusBoth(0, 240, 20, 'preview', true)
%
% Required args:
%   start_point  - Starting x position of stimulus circle (pixels)
%   end_point    - Ending x position of stimulus circle (pixels)
%   stim_size    - Diameter of stimulus circle (pixels)
%
% Returns:
%   stim_log.trial_disparity(i)  - Disparity value for trial i (pixels)
%   stim_log.lead_eye(i)         - Which eye was sent first (1=right, 2=left)
%   stim_log.trial_start(i)      - Start time of trial i (seconds)
%   stim_log.pre_start(i)        - Start of PRE phase
%   stim_log.stim_start(i)       - Start of stimulus video
%   stim_log.stim_end(i)         - End of stimulus video
%   stim_log.post_end(i)         - End of POST phase
%   stim_log.frame_times{i}      - Frame timestamps for trial i

arguments
    start_point  (1,1) double
    end_point    (1,1) double
    stim_size    (1,1) double
    opts.disparities (1,:) double = 0          % Right-eye disparity offsets (px)
    opts.duration    (1,1) double = 6.0        % Stimulus video duration (s)
    opts.frame_rate  (1,1) double = 10         % Video generation frame rate
    opts.rcycles     (1,1) double = 1          % Radial cycles in checkerboard
    opts.tcycles     (1,1) double = 2          % Angular cycles in checkerboard
    opts.window_size (1,1) double = 240        % Psychtoolbox window size for generation
    opts.PORT        (1,1) string = "COM6"     % Right eye serial port
    opts.PORT_left   (1,1) string = "COM5"     % Left eye serial port
    opts.baud        (1,1) double = 115200
    opts.H           (1,1) double = 240
    opts.W           (1,1) double = 240
    opts.n_repeats   (1,1) double = 2          % Repeats per disparity condition
    opts.pre_time    (1,1) double = 2.0        % Gray screen before stim (s)
    opts.post_time   (1,1) double = 3.0        % Gray screen after stim (s)
    opts.target_fps  (1,1) double = 10         % Serial playback FPS
    opts.brightness  (1,1) double = 1.0        % Luminance scale (0-1)
    opts.shuffle     (1,1) logical = true      % Randomize trial order across conditions
    opts.DAQ_flag    (1,1) logical = false
    opts.scope       (1,1) string = '2p'       % 'widefield' or '2p'
    opts.preview     (1,1) logical = false     % Screen preview only, no Teensy
    opts.verbose     (1,1) logical = true
end

n_disp       = length(opts.disparities);
frame_interval = 1 / opts.target_fps;
total_trials   = opts.n_repeats * n_disp;

% === GENERATE VIDEOS ===
if opts.verbose
    fprintf('=== Generating stimulus videos ===\n');
    fprintf('  Left eye:  x=%d -> %d (opposite direction)\n', round(end_point), round(start_point));
    for di = 1:n_disp
        D = opts.disparities(di);
        fprintf('  Right eye: x=%d -> %d (disparity=%+d px)\n', ...
            round(start_point+D), round(end_point+D), round(D));
    end
    fprintf('\n');
end

% Left-eye video (one, opposite direction, fixed)
left_name = sprintf('stim_left_x%d_to_%d_size%d.mp4', ...
    round(end_point), round(start_point), round(stim_size));
if ~isfile(left_name)
    generate_leftright_stim(end_point, start_point, stim_size, ...
        opts.duration, opts.frame_rate, ...
        'output_name', left_name, ...
        'window_size', opts.window_size, ...
        'rcycles', opts.rcycles, ...
        'tcycles', opts.tcycles);
else
    if opts.verbose, fprintf('  Left video already exists, skipping: %s\n\n', left_name); end
end

% Right-eye videos (one per disparity)
right_names = cell(1, n_disp);
for di = 1:n_disp
    D = opts.disparities(di);
    right_names{di} = sprintf('stim_right_x%d_to_%d_size%d_disp%+d.mp4', ...
        round(start_point+D), round(end_point+D), round(stim_size), round(D));
    if ~isfile(right_names{di})
        generate_leftright_stim(start_point+D, end_point+D, stim_size, ...
            opts.duration, opts.frame_rate, ...
            'output_name', right_names{di}, ...
            'window_size', opts.window_size, ...
            'rcycles', opts.rcycles, ...
            'tcycles', opts.tcycles);
    else
        if opts.verbose
            fprintf('  Right video (disp=%+d) already exists, skipping: %s\n\n', round(D), right_names{di});
        end
    end
end

% === PRE-COMPUTE FRAMES ===
if opts.verbose
    fprintf('=== Pre-computing frames ===\n');
end

% Left eye frames (one set)
vL = VideoReader(left_name);
raw_L = {};
vL.CurrentTime = 0;
while hasFrame(vL)
    rgb = readFrame(vL);
    if size(rgb,1) ~= opts.H || size(rgb,2) ~= opts.W
        rgb = imresize(rgb, [opts.H opts.W], 'nearest');
    end
    raw_L{end+1} = rgb; %#ok<AGROW>
end
frames_L   = process_frames(raw_L, 'left', opts.brightness);
payloads_L = cellfun(@rgb565_payload, frames_L, 'UniformOutput', false);
n_frames_L = length(frames_L);
if opts.verbose
    fprintf('  Left eye: %d frames\n', n_frames_L);
end

% Right eye frames per disparity
all_raw_R      = cell(1, n_disp);
all_frames_R   = cell(1, n_disp);
all_payloads_R = cell(1, n_disp);
n_frames_R     = zeros(1, n_disp);
for di = 1:n_disp
    vR = VideoReader(right_names{di});
    raw_R = {};
    vR.CurrentTime = 0;
    while hasFrame(vR)
        rgb = readFrame(vR);
        if size(rgb,1) ~= opts.H || size(rgb,2) ~= opts.W
            rgb = imresize(rgb, [opts.H opts.W], 'nearest');
        end
        raw_R{end+1} = rgb; %#ok<AGROW>
    end
    all_frames_R{di}   = process_frames(raw_R, 'right', opts.brightness);
    all_payloads_R{di} = cellfun(@rgb565_payload, all_frames_R{di}, 'UniformOutput', false);
    n_frames_R(di)     = length(all_frames_R{di});
    if opts.verbose
        fprintf('  Right eye (disp=%+d): %d frames\n', round(opts.disparities(di)), n_frames_R(di));
    end
end

% Gray payloads
gray_img_R   = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), 'right');
gray_img_L   = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), 'left');
gray_payload_R = rgb565_payload(gray_img_R);
gray_payload_L = rgb565_payload(gray_img_L);

% Use first right-eye disparity's duration for timing (all should match)
vid_duration = VideoReader(right_names{1}).Duration;
trial_dur    = opts.pre_time + vid_duration + opts.post_time;
total_dur    = trial_dur * total_trials;

% === TRIAL ORDER ===
% Build condition list: each disparity repeated n_repeats times
cond_list = repelem(1:n_disp, opts.n_repeats);   % [1 1 ... 2 2 ... N N ...]
if opts.shuffle
    rng(0);
    cond_list = cond_list(randperm(total_trials));
end

% Lead-eye alternates every trial: 1=right leads, 2=left leads
lead_seq = mod((0:total_trials-1), 2) + 1;

if opts.verbose
    fprintf('\n=== Teensy Both-Eye Stimulus ===\n');
    fprintf('Disparities: %s px\n', mat2str(opts.disparities));
    fprintf('n_repeats per condition: %d | Total trials: %d\n', opts.n_repeats, total_trials);
    fprintf('Trial structure: PRE %.1fs | STIM %.2fs | POST %.1fs (%.2fs total)\n', ...
        opts.pre_time, vid_duration, opts.post_time, trial_dur);
    fprintf('Total experiment: %.1f s (%.1f min)\n', total_dur, total_dur/60);
    fprintf('Shuffle: %s\n', string(opts.shuffle));
    fprintf('Condition order (disparity px): %s\n', mat2str(opts.disparities(cond_list)));
    fprintf('Lead eye per trial (1=R,2=L): %s\n', mat2str(lead_seq));
    if opts.preview
        fprintf('MODE: PREVIEW (screen only, no Teensy)\n');
    else
        fprintf('Right port: %s | Left port: %s\n', opts.PORT, opts.PORT_left);
    end
    fprintf('================================\n\n');
end

% === SETUP ===
if opts.preview
    figure('Name', 'Both-Eye Preview', 'NumberTitle', 'off');
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
stim_log.parameters      = opts;
stim_log.start_point     = start_point;
stim_log.end_point       = end_point;
stim_log.stim_size       = stim_size;
stim_log.disparities     = opts.disparities;
stim_log.trial_disparity = opts.disparities(cond_list);
stim_log.lead_eye        = lead_seq;
stim_log.trial_start     = zeros(1, total_trials);
stim_log.pre_start       = zeros(1, total_trials);
stim_log.stim_start      = zeros(1, total_trials);
stim_log.stim_end        = zeros(1, total_trials);
stim_log.post_end        = zeros(1, total_trials);
stim_log.frame_times     = cell(1, total_trials);

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
    d_idx  = cond_list(trial);
    D_val  = opts.disparities(d_idx);
    lead   = lead_seq(trial);

    stim_payloads_R = all_payloads_R{d_idx};
    stim_frames_R   = all_frames_R{d_idx};
    nfR = n_frames_R(d_idx);
    nfL = n_frames_L;

    if opts.verbose
        lead_str = 'right'; if lead == 2, lead_str = 'left'; end
        fprintf('--- Trial %d/%d | disparity=%+d px | lead=%s ---\n', ...
            trial, total_trials, round(D_val), lead_str);
    end

    trial_frame_times = [];
    trial_offset = (trial - 1) * trial_dur;

    % --- Helper: send both eyes for one frame ---
    % (inline below per phase for clarity)

    stim_log.trial_start(trial) = toc(experiment_start);

    % === PRE PHASE (gray) ===
    stim_log.pre_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  PRE start:  %.4f s\n', stim_log.pre_start(trial));
    end

    if opts.preview
        set(ih_R, 'CData', gray_img_R); title(ax_R, sprintf('T%d PRE', trial));
        set(ih_L, 'CData', gray_img_L); title(ax_L, sprintf('T%d PRE', trial));
        drawnow;
    end

    frame_count = 0;
    phase_end   = trial_offset + opts.pre_time;
    while toc(experiment_start) < phase_end
        if ~opts.preview
            send_both(s_right, gray_payload_R, s_left, gray_payload_L, opts.W, opts.H, lead);
        end
        frame_count = frame_count + 1;
        trial_frame_times(end+1) = toc(experiment_start); %#ok<AGROW>
        next_frame = trial_offset + frame_count * frame_interval;
        while toc(experiment_start) < next_frame, pause(0.0001); end
    end

    % === STIM PHASE (video) ===
    stim_log.stim_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  STIM start: %.4f s\n', stim_log.stim_start(trial));
    end

    if opts.preview
        title(ax_R, sprintf('T%d STIM (disp=%+d)', trial, round(D_val)));
        title(ax_L, sprintf('T%d STIM (opposite)', trial));
    end

    stim_frame_count = 0;
    stim_offset = trial_offset + opts.pre_time;
    phase_end   = stim_offset + vid_duration;

    while toc(experiment_start) < phase_end
        fi_R = mod(stim_frame_count, nfR) + 1;
        fi_L = mod(stim_frame_count, nfL) + 1;

        if opts.preview
            set(ih_R, 'CData', stim_frames_R{fi_R});
            set(ih_L, 'CData', frames_L{fi_L});
            drawnow;
        else
            send_both(s_right, stim_payloads_R{fi_R}, s_left, payloads_L{fi_L}, opts.W, opts.H, lead);
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
    post_offset    = trial_offset + opts.pre_time + vid_duration;
    phase_end      = trial_offset + trial_dur;
    post_frame_count = 0;

    if opts.preview
        set(ih_R, 'CData', gray_img_R); title(ax_R, sprintf('T%d POST', trial));
        set(ih_L, 'CData', gray_img_L); title(ax_L, sprintf('T%d POST', trial));
        drawnow;
    end

    while toc(experiment_start) < phase_end
        if ~opts.preview
            send_both(s_right, gray_payload_R, s_left, gray_payload_L, opts.W, opts.H, lead);
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

%% --- Helper: send one frame to both eyes with alternating lead ---
function send_both(s_right, payload_R, s_left, payload_L, W, H, lead)
    cmd = sprintf('IMG %d %d', W, H);
    if lead == 1   % right leads
        writeline(s_right, cmd); write(s_right, payload_R, 'uint8');
        writeline(s_left,  cmd); write(s_left,  payload_L, 'uint8');
    else           % left leads
        writeline(s_left,  cmd); write(s_left,  payload_L, 'uint8');
        writeline(s_right, cmd); write(s_right, payload_R, 'uint8');
    end
end

%% --- Frame processing helpers (mirrored from teensyStimulus.m) ---
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
