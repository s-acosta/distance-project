function stim_log = teensyStimulusAngle(left_slope, left_intercept, right_slope, right_intercept, opts)
% Present a binocular sweep specified in VISUAL ANGLE, mapped to per-eye screen
% pixels through independent degree->pixel fits (one slope/intercept per eye).
%
% For each stimulus frame at visual angle `deg`:
%   right_px = right_slope*deg + right_intercept                 (lives in [0, W])
%   left_px  = left_slope*(deg + disparity_deg) + left_intercept (lives in [-W, 0])
%
% The angle sweep is fixed at start_deg -> end_deg (default -30 -> 60). Each eye
% moves at its own pixel rate (its own slope), which is the correct behaviour for
% a common visual target seen through two differently-calibrated screens.
%
% Disparity is specified in DEGREES and applied to the LEFT eye only: it shifts
% the left-eye angle before the deg->pixel conversion, so left frames are
% precomputed per disparity condition; the right eye is shared across conditions.
%
% Left screen is the negative pixel axis, right screen the positive axis; both
% meet at an inner (nasal) edge of 0. Frames are precomputed with make_ball_frame;
% no video files needed. Send order alternates per trial to counterbalance the
% ~30-60 ms serial desync of sequential USB-CDC transfers.
%
% Usage:
%   stim_log = teensyStimulusAngle(0.64, -44.7, 0.88, 64.4)
%   stim_log = teensyStimulusAngle(0.64, -44.7, 0.88, 64.4, 'disparities', [-10 0 10])
%   stim_log = teensyStimulusAngle(0.64, -44.7, 0.88, 64.4, 'preview', true)
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
    opts.disparities   (1,:) double = 0          % LEFT-eye disparity offsets (degrees)
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
    opts.post_time     (1,1) double = 3.0        % Gray screen after stim (s)
    opts.target_fps    (1,1) double = 10         % Serial playback FPS
    opts.brightness    (1,1) double = 1.0        % Luminance scale (0-1)
    opts.bw            (1,1) logical = false     % Black background, solid white ball
    opts.shuffle       (1,1) logical = true      % Randomize trial order across conditions
    opts.DAQ_flag      (1,1) logical = false
    opts.scope         (1,1) string = '2p'       % 'widefield' or '2p'
    opts.preview       (1,1) logical = false     % Screen preview only, no Teensy
    opts.verbose       (1,1) logical = true
end

n_disp          = length(opts.disparities);
frame_interval  = 1 / opts.target_fps;
num_stim_frames = round(opts.stim_duration * opts.target_fps);
stim_duration   = num_stim_frames / opts.target_fps;  % actual duration after rounding
trial_dur       = opts.pre_time + stim_duration + opts.post_time;
total_trials    = opts.n_repeats * n_disp;
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

% Left eye (one array per disparity condition): left_pixel = left_px(deg + D_deg), in [-W, 0]
% Ball is visible when left_pixel in [-W, 0]; pass -left_pixel to make_ball_frame (flips direction)
left_eye_frames_by_disparity   = cell(1, n_disp);
left_eye_payloads_by_disparity = cell(1, n_disp);
for disparity_idx = 1:n_disp
    D_deg = opts.disparities(disparity_idx);
    left_eye_frames_by_disparity{disparity_idx}   = cell(1, num_stim_frames);
    left_eye_payloads_by_disparity{disparity_idx} = cell(1, num_stim_frames);

    for frame_idx = 1:num_stim_frames
        left_pixel_position = left_px(angles(frame_idx) + D_deg);
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
        left_positions = left_px(angles + D_deg);
        n_visible_L    = sum(left_positions >= -opts.W & left_positions <= 0);
        fprintf('  Left eye (disp=%+.1f deg): %d/%d frames visible\n', ...
            D_deg, n_visible_L, num_stim_frames);
    end
end

% === TRIAL ORDER ===
cond_list = repelem(1:n_disp, opts.n_repeats);
if opts.shuffle
    rng(0);
    cond_list = cond_list(randperm(total_trials));
end

% Lead-eye alternates every trial: 1=right leads, 2=left leads
lead_seq = mod((0:total_trials-1), 2) + 1;

if opts.verbose
    fprintf('\n=== Teensy Angle-Based Stimulus ===\n');
    fprintf('Angle sweep: %.1f to %.1f deg\n', opts.start_deg, opts.end_deg);
    fprintf('Left-eye disparities: %s deg\n', mat2str(opts.disparities));
    fprintf('n_repeats per condition: %d | Total trials: %d\n', opts.n_repeats, total_trials);
    fprintf('Trial structure: PRE %.1fs | STIM %.2fs | POST %.1fs (%.2fs total)\n', ...
        opts.pre_time, stim_duration, opts.post_time, trial_dur);
    fprintf('Total experiment: %.1f s (%.1f min)\n', total_dur, total_dur/60);
    fprintf('Shuffle: %s\n', string(opts.shuffle));
    fprintf('Condition order (disparity deg): %s\n', mat2str(opts.disparities(cond_list)));
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
stim_log.left_fit            = [left_slope, left_intercept];
stim_log.right_fit           = [right_slope, right_intercept];
stim_log.start_deg           = opts.start_deg;
stim_log.end_deg             = opts.end_deg;
stim_log.angles              = angles;
stim_log.stim_size           = opts.stim_size;
stim_log.disparities         = opts.disparities;
stim_log.trial_disparity_deg = opts.disparities(cond_list);
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
    D_deg  = opts.disparities(d_idx);
    lead   = lead_seq(trial);

    current_left_payloads = left_eye_payloads_by_disparity{d_idx};
    current_left_frames   = left_eye_frames_by_disparity{d_idx};

    if opts.verbose
        lead_str = 'right'; if lead == 2, lead_str = 'left'; end
        fprintf('--- Trial %d/%d | left disparity=%+.1f deg | lead=%s ---\n', ...
            trial, total_trials, D_deg, lead_str);
    end

    trial_frame_times = [];
    trial_offset = (trial - 1) * trial_dur;

    % Pixel endpoints for this trial
    stim_log.left_eye_start_px(trial)  = left_px(opts.start_deg + D_deg);
    stim_log.left_eye_end_px(trial)    = left_px(opts.end_deg   + D_deg);
    stim_log.right_eye_start_px(trial) = right_px(opts.start_deg);
    stim_log.right_eye_end_px(trial)   = right_px(opts.end_deg);

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
        title(ax_R, sprintf('T%d STIM', trial));
        title(ax_L, sprintf('T%d STIM (disp=%+.1f)', trial, D_deg));
    end

    stim_offset = trial_offset + opts.pre_time;
    for stim_frame_idx = 1:num_stim_frames
        if opts.preview
            set(ih_R, 'CData', rot90(right_eye_frames{stim_frame_idx},      -1));
            set(ih_L, 'CData', rot90(current_left_frames{stim_frame_idx},    1));
            drawnow;
        else
            send_both(s_right, right_eye_payloads{stim_frame_idx}, ...
                      s_left,  current_left_payloads{stim_frame_idx}, ...
                      opts.W, opts.H, lead);
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
