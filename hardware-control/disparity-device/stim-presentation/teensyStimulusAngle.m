function stim_log = teensyStimulusAngle(left_slope, left_intercept, right_slope, right_intercept, opts)
% Present a binocular sweep specified in VISUAL ANGLE, mapped to per-eye screen
% pixels through independent degree->pixel fits (one slope/intercept per eye).
%
% For each stimulus frame at visual angle `deg`:
%   right_px = right_slope*deg + right_intercept                 (lives in [0, W])
%   left_px  = left_slope*(deg - disparity_deg) + left_intercept (lives in [-W, 0])
%
% The angle sweep is fixed at start_deg -> end_deg (default -30 -> 60). Each eye
% moves at its own pixel rate (its own slope), which is the correct behaviour for
% a common visual target seen through two differently-calibrated screens.
%
% Disparity is specified in DEGREES and applied to the LEFT eye only: it shifts
% the left-eye angle before the deg->pixel conversion, so left frames are
% precomputed per disparity condition; the right eye is shared across conditions.
%
% SIGN CONVENTION (matches La Chioma 2019 / primate literature):
%   disparity < 0  -> CROSSED (near): left-eye stimulus shifts NASALLY  (toward pixel 0)
%   disparity > 0  -> UNCROSSED (far): left-eye stimulus shifts TEMPORALLY (toward -W)
% Implemented as left_px(deg - disparity_deg): with left_slope>0 and nasal=pixel 0,
% a negative disparity increases left_px (nasal), a positive disparity decreases it (temporal).
%
% Set opts.monocular=true to instead present ONE eye per trial (the other stays
% gray), alternating eyes like teensyStimulusMonocular. Each eye sweeps its own
% fit's pixel range; disparity is ignored (D=0). Trials = 2*n_repeats (per eye),
% and stim_log.eye_order (1=right, 2=left) records the active eye per trial.
%
% Left screen is the negative pixel axis, right screen the positive axis; both
% meet at an inner (nasal) edge of 0. Frames are precomputed with make_ball_frame;
% no video files needed. Send order alternates per trial to counterbalance the
% ~30-60 ms serial desync of sequential USB-CDC transfers.
%
% Usage:
%   stim_log = teensyStimulusAngle(0.64, -44.7, 0.88, 64.4)
%   stim_log = teensyStimulusAngle(0.64, -44.7, 0.88, 64.4, 'disparities', [-10 0 10])  % snapshot shows automatically
%   stim_log = teensyStimulusAngle(0.64, -44.7, 0.88, 64.4, 'preview', true)
%   stim_log = teensyStimulusAngle(0.64, -44.7, 0.88, 64.4, 'monocular', true)
%
% Required args:
%   left_slope, left_intercept   - left-eye deg->pixel fit
%   right_slope, right_intercept - right-eye deg->pixel fit
%
% Returns:
%   stim_log.trial_disparity_deg(i) - Left-eye disparity for trial i (degrees)
%   stim_log.lead_eye(i)            - Which eye was sent first (1=right, 2=left)
%   stim_log.trial_start(i)         - Start time of trial i (seconds)
%   stim_log.pre_start(i)           - Start of PRE phase
%   stim_log.stim_start(i)          - Start of stimulus
%   stim_log.stim_end(i)            - End of stimulus
%   stim_log.post_end(i)            - End of POST phase
%   stim_log.frame_times{i}         - Frame timestamps for trial i

arguments
    left_slope      (1,1) double
    left_intercept  (1,1) double
    right_slope     (1,1) double
    right_intercept (1,1) double
    opts.start_deg     (1,1) double = -30        % Fixed sweep start (deg)
    opts.end_deg       (1,1) double = 60         % Fixed sweep end (deg)
    opts.stim_size     (1,1) double = 30
    opts.disparities   (1,:) double = 0          % LEFT-eye disparity offsets (degrees); ignored if monocular
    opts.monocular     (1,1) logical = false     % One eye per trial (other gray), alternating eyes; disparity ignored (D=0)
    opts.stim_duration (1,1) double = 6.0        % Stimulus duration (s); num_stim_frames = stim_duration * target_fps
    opts.rcycles       (1,1) double = 1          % Radial cycles in checkerboard
    opts.tcycles       (1,1) double = 2          % Angular cycles in checkerboard
    opts.PORT          (1,1) string = "COM6"     % Right eye serial port
    opts.PORT_left     (1,1) string = "COM5"     % Left eye serial port
    opts.baud          (1,1) double = 115200
    opts.H             (1,1) double = 240
    opts.W             (1,1) double = 240
    opts.n_repeats     (1,1) double = 12          % Repeats per disparity condition
    opts.pre_time      (1,1) double = 2.0        % Gray screen before stim (s)
    opts.post_time     (1,1) double = 1.0        % Gray screen after stim (s)
    opts.target_fps    (1,1) double = 10         % Serial playback FPS
    opts.brightness    (1,1) double = 0.6        % Luminance scale (0-1)
    opts.bw            (1,1) logical = false     % Black background, solid white ball
    opts.shuffle       (1,1) logical = true      % Randomize trial order across conditions
    opts.DAQ_flag      (1,1) logical = false
    opts.scope         (1,1) string = '2p'       % 'widefield' or '2p'
    opts.preview       (1,1) logical = false     % Screen preview only, no Teensy
    opts.snapshot       (1,1) logical = true     % Show a per-disparity snapshot montage at startup (binocular w/ disparity)
    opts.snapshot_angle (1,1) double  = 0        % Visual angle (deg) for the snapshot; uses nearest swept frame
    opts.snapshot_tick_deg (1,1) double = 30     % Spacing (deg) of the visual-angle ruler drawn on the snapshot
    opts.verbose       (1,1) logical = true
end

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

% Visual angle for each stimulus frame (same for every trial)
angles = linspace(opts.start_deg, opts.end_deg, num_stim_frames);

% deg -> pixel maps (right eye has no disparity; left eye disparity added at use)
right_px = @(deg) right_slope * deg + right_intercept;
left_px  = @(deg) left_slope  * deg + left_intercept;

% === PRE-COMPUTE FRAMES ===
if opts.verbose
    fprintf('=== Pre-computing frames ===\n');
    fprintf('  Angle sweep: %.1f to %.1f deg | %d frames at %d fps (%.1fs)\n', ...
        opts.start_deg, opts.end_deg, num_stim_frames, opts.target_fps, stim_duration);
    fprintf('  Fits: left px = %.3f*deg %+.2f | right px = %.3f*deg %+.2f\n', ...
        left_slope, left_intercept, right_slope, right_intercept);
end

% Gray images (rotated per eye for consistency with ball frames)
gray_img_R     = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), 'right');
gray_img_L     = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), 'left');
gray_payload_R = rgb565_payload(gray_img_R);
gray_payload_L = rgb565_payload(gray_img_L);

% Right eye (shared across disparity conditions): right_pixel = right_px(deg), in [0, W]
right_eye_frames   = cell(1, num_stim_frames);
right_eye_payloads = cell(1, num_stim_frames);
for frame_idx = 1:num_stim_frames
    right_pixel_position = right_px(angles(frame_idx));
    if right_pixel_position >= 0 && right_pixel_position <= opts.W
        ball_frame = make_ball_frame(right_pixel_position, opts.stim_size, ...
                                     opts.W, opts.H, opts.rcycles, opts.tcycles, ...
                                     opts.bw, opts.brightness);
        right_eye_frames{frame_idx}   = rotate_for_eye(ball_frame, 'right');
        right_eye_payloads{frame_idx} = rgb565_payload(right_eye_frames{frame_idx});
    else
        right_eye_frames{frame_idx}   = gray_img_R;
        right_eye_payloads{frame_idx} = gray_payload_R;
    end
end

if opts.verbose
    right_positions = right_px(angles);
    n_visible_R = sum(right_positions >= 0 & right_positions <= opts.W);
    fprintf('  Right eye: %d/%d frames visible\n', n_visible_R, num_stim_frames);
end

% Left eye (one array per disparity condition): left_pixel = left_px(deg - D_deg), in [-W, 0]
% Ball is visible when left_pixel in [-W, 0]; pass -left_pixel to make_ball_frame (flips direction)
% Monocular ignores disparity, so only the D=0 set is built (index 1).
if opts.monocular
    left_disp_list = 0;
else
    left_disp_list = opts.disparities;
end
n_left_sets = numel(left_disp_list);
left_eye_frames_by_disparity   = cell(1, n_left_sets);
left_eye_payloads_by_disparity = cell(1, n_left_sets);
for disparity_idx = 1:n_left_sets
    D_deg = left_disp_list(disparity_idx);
    left_eye_frames_by_disparity{disparity_idx}   = cell(1, num_stim_frames);
    left_eye_payloads_by_disparity{disparity_idx} = cell(1, num_stim_frames);

    for frame_idx = 1:num_stim_frames
        left_pixel_position = left_px(angles(frame_idx) - D_deg);
        if left_pixel_position >= -opts.W && left_pixel_position <= 0
            ball_frame = make_ball_frame(-left_pixel_position, opts.stim_size, ...
                                         opts.W, opts.H, opts.rcycles, opts.tcycles, ...
                                         opts.bw, opts.brightness);
            left_eye_frames_by_disparity{disparity_idx}{frame_idx}   = rotate_for_eye(ball_frame, 'left');
            left_eye_payloads_by_disparity{disparity_idx}{frame_idx} = ...
                rgb565_payload(left_eye_frames_by_disparity{disparity_idx}{frame_idx});
        else
            left_eye_frames_by_disparity{disparity_idx}{frame_idx}   = gray_img_L;
            left_eye_payloads_by_disparity{disparity_idx}{frame_idx} = gray_payload_L;
        end
    end

    if opts.verbose
        left_positions = left_px(angles - D_deg);
        n_visible_L    = sum(left_positions >= -opts.W & left_positions <= 0);
        fprintf('  Left eye (disp=%+.1f deg): %d/%d frames visible\n', ...
            D_deg, n_visible_L, num_stim_frames);
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
    fprintf('\n=== Teensy Angle-Based Stimulus ===\n');
    if opts.monocular
        fprintf('Mode: MONOCULAR (one eye per trial, disparity ignored)\n');
    else
        fprintf('Mode: BINOCULAR\n');
    end
    fprintf('Angle sweep: %.1f to %.1f deg\n', opts.start_deg, opts.end_deg);
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
    figure('Name', 'Angle-Based Preview', 'NumberTitle', 'off');
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
stim_log.parameters          = opts;
if opts.monocular
    stim_log.mode = 'monocular';
else
    stim_log.mode = 'binocular';
end
stim_log.left_fit            = [left_slope, left_intercept];
stim_log.right_fit           = [right_slope, right_intercept];
stim_log.start_deg           = opts.start_deg;
stim_log.end_deg             = opts.end_deg;
stim_log.angles              = angles;
stim_log.stim_size           = opts.stim_size;
stim_log.disparities         = opts.disparities;
if opts.monocular
    stim_log.eye_order        = eye_seq;                  % 1=right active, 2=left active
    stim_log.trial_disparity_deg = zeros(1, total_trials);
else
    stim_log.trial_disparity_deg = opts.disparities(cond_list);
end
stim_log.lead_eye            = lead_seq;
stim_log.trial_start         = zeros(1, total_trials);
stim_log.pre_start           = zeros(1, total_trials);
stim_log.stim_start          = zeros(1, total_trials);
stim_log.stim_end            = zeros(1, total_trials);
stim_log.post_end            = zeros(1, total_trials);
stim_log.frame_times         = cell(1, total_trials);
stim_log.left_eye_start_px   = nan(1, total_trials);
stim_log.left_eye_end_px     = nan(1, total_trials);
stim_log.right_eye_start_px  = nan(1, total_trials);
stim_log.right_eye_end_px    = nan(1, total_trials);

% === DISPARITY SNAPSHOT (sanity check) ===
% One-time montage at angle ~= snapshot_angle: for each disparity condition, show the actual
% rendered left/right eye frames. Right eye is shared across conditions, so its column is identical
% every row while the left ball shifts with disparity -- that is the thing to eyeball. Reuses the
% precomputed frames, so no hardware and no re-rendering.
show_snapshot = opts.snapshot && ~opts.monocular && ...
                (numel(opts.disparities) > 1 || any(opts.disparities ~= 0));
if show_snapshot
    [~, snap_idx] = min(abs(angles - opts.snapshot_angle));
    snap_angle    = angles(snap_idx);            % actual nearest sampled angle
    frame_R       = right_eye_frames{snap_idx};  % right eye shared across all disparities
    rpx           = right_px(snap_angle);

    % Fixed per-eye visual-angle ruler: tick column for angle a is where that eye's fit renders a
    % (right: right_px(a); left displayed col: W+1+left_px(a) after the display rot90). The ruler is
    % independent of disparity, so the right ball stays on its tick while the left ball slides along
    % the ticks as D changes -> disparity = (tick under left ball) - (tick under right ball).
    tick_step   = opts.snapshot_tick_deg;
    tick_angles = ceil(opts.start_deg/tick_step)*tick_step : tick_step : floor(opts.end_deg/tick_step)*tick_step;
    R_cols = right_px(tick_angles);              vR = R_cols >= 0.5 & R_cols <= opts.W + 0.5;
    L_cols = opts.W + 1 + left_px(tick_angles);  vL = L_cols >= 0.5 & L_cols <= opts.W + 0.5;

    figure('Name', 'Disparity Snapshot', 'NumberTitle', 'off');
    for k = 1:n_disp
        D       = opts.disparities(k);
        frame_L = left_eye_frames_by_disparity{k}{snap_idx};
        lpx     = left_px(snap_angle - D);
        if lpx < -opts.W || lpx > 0, off_lbl = ' (off)'; else, off_lbl = ''; end
        if D < 0, sense = ' crossed/near'; elseif D > 0, sense = ' uncrossed/far'; else, sense = ' zero'; end

        ax = subplot(n_disp, 2, (k-1)*2 + 1);
        imshow(rot90(frame_L, 1), 'Parent', ax);
        draw_angle_ruler(ax, L_cols(vL), tick_angles(vL), opts.H);
        title(ax, sprintf('D=%+.1f%s  Left px=%.1f%s', D, sense, lpx, off_lbl));

        ax = subplot(n_disp, 2, (k-1)*2 + 2);
        imshow(rot90(frame_R, -1), 'Parent', ax);
        draw_angle_ruler(ax, R_cols(vR), tick_angles(vR), opts.H);
        title(ax, sprintf('Right px=%.1f', rpx));
    end
    sgtitle(sprintf('Disparity snapshots @ angle \\approx %.1f\\circ  |  ticks = visual angle (deg)  |  D<0 crossed (left nasal), D>0 uncrossed (left temporal)', snap_angle));
    drawnow;
end

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

    % Which eyes are active this trial, and the left-eye disparity in play
    if opts.monocular
        active_eye = eye_seq(trial);        % 1=right, 2=left
        right_on   = (active_eye == 1);
        left_on    = (active_eye == 2);
        D_deg      = 0;                      % disparity ignored in monocular
    else
        right_on   = true;
        left_on    = true;
        D_deg      = opts.disparities(d_idx);
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
            fprintf('--- Trial %d/%d | left disparity=%+.1f deg | lead=%s ---\n', ...
                trial, total_trials, D_deg, lead_str);
        end
    end

    trial_frame_times = [];
    trial_offset = (trial - 1) * trial_dur;

    % Pixel endpoints for this trial (idle eye in monocular stays NaN)
    if left_on
        stim_log.left_eye_start_px(trial) = left_px(opts.start_deg - D_deg);
        stim_log.left_eye_end_px(trial)   = left_px(opts.end_deg   - D_deg);
    end
    if right_on
        stim_log.right_eye_start_px(trial) = right_px(opts.start_deg);
        stim_log.right_eye_end_px(trial)   = right_px(opts.end_deg);
    end

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
        % Active eye shows the ball; idle eye (monocular) stays gray
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

%% --- Ball frame generation ---
function frame = make_ball_frame(x_pos, stim_size, W, H, rcycles, tcycles, bw, brightness)
% Build ball at position (x_pos, H/2).
% bw=true: solid white ball on black background.
% bw=false: radial checkerboard on gray background.
% brightness: global luminance scale (0=black, 1=full brightness).

    xylim = pi * rcycles;
    [xg, yg] = meshgrid(linspace(-xylim, xylim, stim_size));
    circle = (xg.^2 + yg.^2) <= xylim^2;
    if bw
        ball = double(circle);
    else
        at = atan2(yg, xg);
        checksN = (1 + sign(sin(at * tcycles) + eps) .* sign(sin(sqrt(xg.^2 + yg.^2)))) / 2;
        ball = circle .* checksN + 0.5 * ~circle;
    end
    ball_rgb = uint8(round(ball * 255 * brightness));
    ball_rgb = cat(3, ball_rgb, ball_rgb, ball_rgb);

    if bw
        frame = uint8(zeros(H, W, 3));
    else
        frame = uint8(ones(H, W, 3) * round(127 * brightness));
    end
    y_center = round(H / 2);
    half     = floor(stim_size / 2);

    y1 = max(1, y_center - half);
    y2 = min(H, y_center - half + stim_size - 1);
    x1 = max(1, round(x_pos) - half);
    x2 = min(W, round(x_pos) - half + stim_size - 1);

    by1 = y1 - (y_center - half) + 1;
    by2 = by1 + (y2 - y1);
    bx1 = x1 - (round(x_pos) - half) + 1;
    bx2 = bx1 + (x2 - x1);

    frame(y1:y2, x1:x2, :) = ball_rgb(by1:by2, bx1:bx2, :);
end

%% --- Angle ruler overlay for the disparity snapshot ---
function draw_angle_ruler(ax, cols, labels_deg, H)
% Draw dashed vertical guide lines at the given displayed columns (one per reference visual angle),
% spanning the full image height, and label them with readable numeric ticks along the x-axis
% (imshow turns the axis off, so re-enable just the x-axis). cols must be ascending.
    hold(ax, 'on');
    for t = 1:numel(cols)
        plot(ax, [cols(t) cols(t)], [0.5, H + 0.5], '--', 'Color', [1 1 0], 'LineWidth', 0.5);
    end
    hold(ax, 'off');
    if isempty(cols), return; end
    ax.XAxis.Visible = 'on';
    ax.XTick         = cols;
    ax.XTickLabel    = string(labels_deg);
    ax.YTick         = [];
    ax.TickLength    = [0 0];
    ax.FontSize      = 7;
    ax.XColor        = [0.35 0.35 0.35];
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
