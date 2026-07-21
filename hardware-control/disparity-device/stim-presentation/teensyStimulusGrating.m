function stim_log = teensyStimulusGrating(opts)
% Present a binocular full-field drifting vertical grating with a PHASE
% disparity between the two eyes, mapped through a hardcoded, hardware-
% measured deg->pixel slope (same value for both eyes, see EYE_SLOPE below).
%
% Both eyes drift the same vertical sinusoidal grating at the same temporal
% frequency (opts.temp_freq). The grating's spatial period is set directly in
% SCREEN PIXELS (opts.stim_size, one white half-cycle + one black half-cycle),
% matching the known on-screen footprint of the ball stimulus, rather than a
% period derived from a theoretical degrees-based spatial frequency:
%   cpp_eye = 1 / stim_size   (cycles/pixel, same value on both screens)
%
% Disparity is specified in DEGREES OF VISUAL ANGLE (same convention as
% teensyStimulusAngle's `disparities`) and applied to the LEFT eye only as a
% static phase offset for the whole trial. Since the grating period is in
% pixels but the disparity is in degrees, EYE_SLOPE (px/deg) is used here to
% convert the disparity to a pixel shift before turning it into a phase:
%   phase_offset_deg = disparity_deg * EYE_SLOPE / stim_size * 360
%
% Orientation is fixed at vertical (grating varies along the horizontal/x
% axis) since horizontal interocular disparity only produces a meaningful
% phase shift for gratings whose luminance varies along x.
%
% Set opts.monocular=true to instead present ONE eye per trial (the other
% stays gray), alternating eyes like teensyStimulusMonocular. Disparity is
% ignored (D=0). Trials = 2*n_repeats (per eye), and stim_log.eye_order
% (1=right, 2=left) records the active eye per trial.
%
% Frames are precomputed per disparity condition; no video files needed.
% Send order alternates per trial to counterbalance the ~30-60 ms serial
% desync of sequential USB-CDC transfers.
%
% Usage:
%   stim_log = teensyStimulusGrating()
%   stim_log = teensyStimulusGrating('disparities', [-10 0 10])
%   stim_log = teensyStimulusGrating('stim_size', 30, 'temp_freq', 2)
%   stim_log = teensyStimulusGrating('preview', true)
%
% Eye calibration: both eyes share a hardcoded, hardware-measured deg->pixel
% slope of 0.76 px/deg (EYE_SLOPE below), used only to convert the
% degrees-based disparity into a phase shift (see above); no intercept is
% needed since a full-field grating has no absolute position anchor.
%
% Returns:
%   stim_log.trial_disparity_deg(i)    - Left-eye disparity for trial i (deg visual angle)
%   stim_log.trial_phase_offset_deg(i) - Left-eye phase offset for trial i (deg)
%   stim_log.lead_eye(i)               - Which eye was sent first (1=right, 2=left)
%   stim_log.trial_start(i)            - Start time of trial i (seconds)
%   stim_log.pre_start(i)              - Start of PRE phase
%   stim_log.stim_start(i)             - Start of stimulus
%   stim_log.stim_end(i)               - End of stimulus
%   stim_log.post_end(i)               - End of POST phase
%   stim_log.frame_times{i}            - Frame timestamps for trial i

arguments
    opts.stim_size      (1,1) double = 30          % Grating period in screen pixels (one white + one black half-cycle) - same convention as the ball's stim_size
    opts.temp_freq      (1,1) double = 2.0        % Drift speed (Hz)
    opts.contrast        (1,1) double = 1.0        % Michelson contrast (0-1)
    opts.phase0          (1,1) double = 0          % Starting phase (deg), shared by both eyes
    opts.disparities     (1,:) double = 0          % LEFT-eye disparity (degrees visual angle); ignored if monocular
    opts.monocular       (1,1) logical = false      % One eye per trial (other gray), alternating eyes; disparity ignored (D=0)
    opts.stim_duration   (1,1) double = 3.0        % Stimulus duration (s); num_stim_frames = stim_duration * target_fps. Matches La Chioma et al. 2020 (J Neurosci) mouse V1/LM/RL phase-disparity protocol
    opts.PORT            (1,1) string = "COM6"      % Right eye serial port
    opts.PORT_left       (1,1) string = "COM5"      % Left eye serial port
    opts.baud            (1,1) double = 115200
    opts.H               (1,1) double = 240
    opts.W               (1,1) double = 240
    opts.n_repeats       (1,1) double = 12          % Repeats per disparity condition
    opts.pre_time        (1,1) double = 1.0        % Gray screen before stim (s); La Chioma et al. 2020 use a 2s ITI, split here as pre+post
    opts.post_time       (1,1) double = 1.0        % Gray screen after stim (s); see pre_time
    opts.target_fps      (1,1) double = 10          % Serial playback FPS
    opts.brightness      (1,1) double = 0.6        % Luminance scale (0-1)
    opts.shuffle         (1,1) logical = true       % Randomize trial order across conditions
    opts.DAQ_flag        (1,1) logical = false
    opts.scope           (1,1) string = '2p'        % 'widefield' or '2p'
    opts.preview         (1,1) logical = false      % Screen preview only, no Teensy
    opts.verbose         (1,1) logical = true
end

% Hardware-measured deg->pixel slope, shared by both eyes (no intercept
% needed: a full-field grating has no absolute position anchor, only the
% local px/deg density matters for rendering spatial frequency).
EYE_SLOPE   = 0.76;   % px/deg
left_slope  = EYE_SLOPE;
right_slope = EYE_SLOPE;

n_disp          = length(opts.disparities);
frame_interval  = 1 / opts.target_fps;
num_stim_frames = round(opts.stim_duration * opts.target_fps);
stim_duration   = num_stim_frames / opts.target_fps;  % actual duration after rounding
trial_dur       = opts.pre_time + stim_duration + opts.post_time;
if opts.monocular
    total_trials = 2 * opts.n_repeats;   % n_repeats per eye
else
    total_trials = opts.n_repeats * n_disp;
end
total_dur       = trial_dur * total_trials;

% Screen-pixel spatial frequency (cycles/pixel), set directly from the known
% pixel period (same value on both screens; no per-eye slope conversion needed
% here since we're specifying pixels, not a degrees-based frequency)
cpp_right = 1 / opts.stim_size;
cpp_left  = 1 / opts.stim_size;

% Sign of the temporal term that actually renders as on-screen drift direction
% (negated here so opts.temp_freq > 0 drifts rightward; flip this to reverse it)
drift_freq = -opts.temp_freq;

% === PRE-COMPUTE FRAMES ===
if opts.verbose
    fprintf('=== Pre-computing frames ===\n');
    fprintf('  Grating period: %d px/cycle (%.4f cyc/px) | Temporal freq: %.2f Hz | Contrast: %.2f\n', ...
        opts.stim_size, cpp_right, opts.temp_freq, opts.contrast);
    fprintf('  %d frames at %d fps (%.1fs)\n', num_stim_frames, opts.target_fps, stim_duration);
end

% Gray images (rotated per eye for consistency with grating frames)
gray_img_R     = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), 'right');
gray_img_L     = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), 'left');
gray_payload_R = rgb565_payload(gray_img_R);
gray_payload_L = rgb565_payload(gray_img_L);

% Right eye (shared across disparity conditions): phase offset always 0
right_eye_frames   = cell(1, num_stim_frames);
right_eye_payloads = cell(1, num_stim_frames);
for frame_idx = 1:num_stim_frames
    t_sec = (frame_idx - 1) / opts.target_fps;
    grating_frame = make_grating_frame(t_sec, cpp_right, drift_freq, opts.phase0, 0, ...
        opts.contrast, opts.W, opts.H, opts.brightness);
    right_eye_frames{frame_idx}   = rotate_for_eye(grating_frame, 'right');
    right_eye_payloads{frame_idx} = rgb565_payload(right_eye_frames{frame_idx});
end

% Left eye (one array per disparity condition): static phase offset for the trial.
% Monocular ignores disparity, so only the D=0 set is built (index 1).
if opts.monocular
    left_disp_list = 0;
else
    left_disp_list = opts.disparities;
end
n_left_sets = numel(left_disp_list);
left_eye_frames_by_disparity   = cell(1, n_left_sets);
left_eye_payloads_by_disparity = cell(1, n_left_sets);
phase_offset_by_disparity      = zeros(1, n_left_sets);
for disparity_idx = 1:n_left_sets
    D_deg = left_disp_list(disparity_idx);
    phase_offset_deg = D_deg * EYE_SLOPE / opts.stim_size * 360;
    phase_offset_by_disparity(disparity_idx) = phase_offset_deg;

    left_eye_frames_by_disparity{disparity_idx}   = cell(1, num_stim_frames);
    left_eye_payloads_by_disparity{disparity_idx} = cell(1, num_stim_frames);

    for frame_idx = 1:num_stim_frames
        t_sec = (frame_idx - 1) / opts.target_fps;
        % Negate cpp for the left eye only: the two screens are mounted
        % mirror-symmetric, and rotate_for_eye only mirrors the right eye's
        % frame, so the left eye's spatial coordinate must be pre-mirrored
        % here for both eyes to drift the same real-world direction (same
        % convention as the "-left_pixel_position" flip in teensyStimulusAngle).
        grating_frame = make_grating_frame(t_sec, -cpp_left, drift_freq, opts.phase0, phase_offset_deg, ...
            opts.contrast, opts.W, opts.H, opts.brightness);
        left_eye_frames_by_disparity{disparity_idx}{frame_idx}   = rotate_for_eye(grating_frame, 'left');
        left_eye_payloads_by_disparity{disparity_idx}{frame_idx} = ...
            rgb565_payload(left_eye_frames_by_disparity{disparity_idx}{frame_idx});
    end

    if opts.verbose
        fprintf('  Left eye (disp=%+.1f deg -> phase=%+.1f deg): %d frames\n', ...
            D_deg, phase_offset_deg, num_stim_frames);
    end
end

% === TRIAL ORDER ===
if opts.monocular
    % Alternating pairs [R,L],[L,R],... so each eye gets n_repeats trials,
    % never the same eye more than twice in a row (1=right active, 2=left active)
    eye_seq = zeros(1, total_trials);
    for p = 1:opts.n_repeats
        idx = (p-1)*2 + 1;
        if mod(p, 2) == 1
            eye_seq(idx:idx+1) = [1, 2];
        else
            eye_seq(idx:idx+1) = [2, 1];
        end
    end
    cond_list = ones(1, total_trials);   % always the D=0 left set (index 1)
else
    cond_list = repelem(1:n_disp, opts.n_repeats);
    if opts.shuffle
        rng(0);
        cond_list = cond_list(randperm(total_trials));
    end
end

% Lead-eye alternates every trial: 1=right leads, 2=left leads
lead_seq = mod((0:total_trials-1), 2) + 1;

if opts.verbose
    fprintf('\n=== Teensy Grating Phase-Disparity Stimulus ===\n');
    if opts.monocular
        fprintf('Mode: MONOCULAR (one eye per trial, disparity ignored)\n');
    else
        fprintf('Mode: BINOCULAR\n');
    end
    if opts.monocular
        fprintf('n_repeats per eye: %d | Total trials: %d\n', opts.n_repeats, total_trials);
        fprintf('Active eye per trial (1=R,2=L): %s\n', mat2str(eye_seq));
    else
        fprintf('Left-eye disparities: %s deg\n', mat2str(opts.disparities));
        fprintf('n_repeats per condition: %d | Total trials: %d\n', opts.n_repeats, total_trials);
    end
    fprintf('Trial structure: PRE %.1fs | STIM %.2fs | POST %.1fs (%.2fs total)\n', ...
        opts.pre_time, stim_duration, opts.post_time, trial_dur);
    fprintf('Total experiment: %.1f s (%.1f min)\n', total_dur, total_dur/60);
    fprintf('Shuffle: %s\n', string(opts.shuffle));
    if ~opts.monocular
        fprintf('Condition order (disparity deg): %s\n', mat2str(opts.disparities(cond_list)));
    end
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
    figure('Name', 'Grating Disparity Preview', 'NumberTitle', 'off');
    ax_L = subplot(1,2,1); imshow(rot90(gray_img_L,  1), 'Parent', ax_L); title(ax_L, 'Left eye');
    ax_R = subplot(1,2,2); imshow(rot90(gray_img_R, -1), 'Parent', ax_R); title(ax_R, 'Right eye');
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
stim_log.parameters   = opts;
if opts.monocular
    stim_log.mode = 'monocular';
else
    stim_log.mode = 'binocular';
end
stim_log.left_slope     = left_slope;
stim_log.right_slope    = right_slope;
stim_log.stim_size      = opts.stim_size;
stim_log.temp_freq      = opts.temp_freq;
stim_log.contrast       = opts.contrast;
stim_log.phase0         = opts.phase0;
stim_log.disparities    = opts.disparities;
if opts.monocular
    stim_log.eye_order              = eye_seq;   % 1=right active, 2=left active
    stim_log.trial_disparity_deg    = zeros(1, total_trials);
    stim_log.trial_phase_offset_deg = zeros(1, total_trials);
else
    stim_log.trial_disparity_deg    = opts.disparities(cond_list);
    stim_log.trial_phase_offset_deg = phase_offset_by_disparity(cond_list);
end
stim_log.lead_eye     = lead_seq;
stim_log.trial_start  = zeros(1, total_trials);
stim_log.pre_start    = zeros(1, total_trials);
stim_log.stim_start   = zeros(1, total_trials);
stim_log.stim_end     = zeros(1, total_trials);
stim_log.post_end     = zeros(1, total_trials);
stim_log.frame_times  = cell(1, total_trials);

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
    lead   = lead_seq(trial);

    % Which eyes are active this trial, and the left-eye disparity/phase in play
    if opts.monocular
        active_eye = eye_seq(trial);        % 1=right, 2=left
        right_on   = (active_eye == 1);
        left_on    = (active_eye == 2);
        D_deg      = 0;                      % disparity ignored in monocular
        phase_offset_deg = 0;
    else
        right_on   = true;
        left_on    = true;
        D_deg      = opts.disparities(d_idx);
        phase_offset_deg = phase_offset_by_disparity(d_idx);
    end

    % Per-eye frame/payload sources for the STIM phase (used only when the eye is on)
    current_right_payloads = right_eye_payloads;
    current_right_frames   = right_eye_frames;
    current_left_payloads  = left_eye_payloads_by_disparity{d_idx};
    current_left_frames    = left_eye_frames_by_disparity{d_idx};

    if opts.verbose
        lead_str = 'right'; if lead == 2, lead_str = 'left'; end
        if opts.monocular
            eye_str = 'right'; if left_on, eye_str = 'left'; end
            fprintf('--- Trial %d/%d | active eye=%s | lead=%s ---\n', ...
                trial, total_trials, eye_str, lead_str);
        else
            fprintf('--- Trial %d/%d | left disparity=%+.1f deg (phase=%+.1f deg) | lead=%s ---\n', ...
                trial, total_trials, D_deg, phase_offset_deg, lead_str);
        end
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
        set(ih_R, 'CData', rot90(gray_img_R, -1)); title(ax_R, sprintf('T%d PRE', trial));
        set(ih_L, 'CData', rot90(gray_img_L,  1)); title(ax_L, sprintf('T%d PRE', trial));
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

    % === STIM PHASE ===
    stim_log.stim_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  STIM start: %.4f s\n', stim_log.stim_start(trial));
    end

    if opts.preview
        if opts.monocular
            if right_on, r_lbl = ' (active)'; else, r_lbl = ' (idle)'; end
            if left_on,  l_lbl = ' (active)'; else, l_lbl = ' (idle)'; end
            title(ax_R, sprintf('T%d STIM%s', trial, r_lbl));
            title(ax_L, sprintf('T%d STIM%s', trial, l_lbl));
        else
            title(ax_R, sprintf('T%d STIM', trial));
            title(ax_L, sprintf('T%d STIM (disp=%+.1f)', trial, D_deg));
        end
    end

    stim_offset = trial_offset + opts.pre_time;
    for stim_frame_idx = 1:num_stim_frames
        % Active eye shows the grating; idle eye (monocular) stays gray
        if right_on
            frame_R = current_right_frames{stim_frame_idx};   payload_R = current_right_payloads{stim_frame_idx};
        else
            frame_R = gray_img_R;                             payload_R = gray_payload_R;
        end
        if left_on
            frame_L = current_left_frames{stim_frame_idx};    payload_L = current_left_payloads{stim_frame_idx};
        else
            frame_L = gray_img_L;                             payload_L = gray_payload_L;
        end

        if opts.preview
            set(ih_R, 'CData', rot90(frame_R, -1));
            set(ih_L, 'CData', rot90(frame_L,  1));
            drawnow;
        else
            send_both(s_right, payload_R, s_left, payload_L, opts.W, opts.H, lead);
        end

        trial_frame_times(end+1) = toc(experiment_start); %#ok<AGROW>
        target_time = stim_offset + stim_frame_idx * frame_interval;
        while toc(experiment_start) < target_time, pause(0.0001); end
    end

    stim_log.stim_end(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  STIM end:   %.4f s (duration: %.4f s)\n', ...
            stim_log.stim_end(trial), stim_log.stim_end(trial) - stim_log.stim_start(trial));
    end

    % === POST PHASE (gray) ===
    post_offset      = trial_offset + opts.pre_time + stim_duration;
    phase_end        = trial_offset + trial_dur;
    post_frame_count = 0;

    if opts.preview
        set(ih_R, 'CData', rot90(gray_img_R, -1)); title(ax_R, sprintf('T%d POST', trial));
        set(ih_L, 'CData', rot90(gray_img_L,  1)); title(ax_L, sprintf('T%d POST', trial));
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
    set(ih_R, 'CData', rot90(gray_img_R, -1)); title(ax_R, 'Done');
    set(ih_L, 'CData', rot90(gray_img_L,  1)); title(ax_L, 'Done');
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

%% --- Vertical drifting sinusoidal grating frame generation ---
function frame = make_grating_frame(t_sec, cpp, temp_freq, phase0_deg, phase_offset_deg, contrast, W, H, brightness)
% Full-field vertical grating: luminance varies along x (columns) only, so
% bars run vertically and drift horizontally as phase advances with t_sec.
%   cpp:               screen spatial frequency for this eye (cycles/pixel)
%   phase0_deg:        shared starting phase (deg)
%   phase_offset_deg:  static per-trial disparity phase (deg), 0 for right eye

    xg = repmat((0:W-1) - (W-1)/2, H, 1);
    arg = 2*pi*cpp*xg + 2*pi*temp_freq*t_sec + deg2rad(phase0_deg + phase_offset_deg);
    L = 0.5 + (contrast/2) * sin(arg);
    gray = uint8(round(255 * brightness * L));
    frame = cat(3, gray, gray, gray);
end

%% --- Frame rotation for each eye ---
function frame = rotate_for_eye(frame, eye)
    if strcmp(eye, 'right')
        frame = flip(frame, 2);
        frame = rot90(frame, -1);
    elseif strcmp(eye, 'left')
        frame = rot90(frame, 1);
    end
end

%% --- RGB565 serialization ---
function payload = rgb565_payload(img)
    if ~isa(img, 'uint8'), img = im2uint8(img); end
    R5 = bitshift(uint16(img(:,:,1)), -3);
    G6 = bitshift(uint16(img(:,:,2)), -2);
    B5 = bitshift(uint16(img(:,:,3)), -3);
    rgb565 = bitor(bitshift(R5,11), bitor(bitshift(G6,5), B5));
    payload = typecast(reshape(rgb565.', [], 1), 'uint8');
end
