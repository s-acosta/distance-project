function stim_log = teensyStimulus(video_path, opts)
% Play video on Teensy with trial structure and precise timing control
%
% Usage:
%   stim_log = teensyStimulus('stim.mp4')
%   stim_log = teensyStimulus('stim.mp4', 'n_repeats', 5, 'pre_time', 2, 'post_time', 2)
%   stim_log = teensyStimulus('stim.mp4', 'preview', true)
%   stim_log = teensyStimulus('stim.mp4', 'brightness', 0.5, 'eye', 'left')
%   stim_log = teensyStimulus('stim.mp4', 'eye', 'both', 'PORT', 'COM3', 'PORT_left', 'COM4')
%
% eye options:
%   "right"  - single right-eye Teensy (PORT)
%   "left"   - single left-eye Teensy  (PORT)
%   "none"   - single Teensy, no rotation
%   "both"   - two Teensys; each trial one eye semi-randomly gets the stimulus,
%              the other shows gray. PORT = right eye, PORT_left = left eye.
%
% Returns:
%   stim_log.trial_start(i)   - Start time of trial i (seconds)
%   stim_log.pre_start(i)     - Start of PRE phase (gray screen)
%   stim_log.stim_start(i)    - Start of stimulus video
%   stim_log.stim_end(i)      - End of stimulus video
%   stim_log.post_end(i)      - End of POST phase (gray screen)
%   stim_log.frame_times{i}   - Frame timestamps for trial i
%   stim_log.eye_order         - (both mode only) 1=right, 2=left per trial

arguments
    video_path (1,1) string
    opts.PORT (1,1) string = "COM3"
    opts.PORT_left (1,1) string = "COM4"  % Left eye port (only used when eye="both")
    opts.baud (1,1) double = 115200
    opts.H (1,1) double = 240
    opts.W (1,1) double = 240
    opts.n_repeats (1,1) double = 2       % Number of trial repetitions
    opts.pre_time (1,1) double = 2.0      % Gray screen before stimulus (seconds)
    opts.post_time (1,1) double = 2.0     % Gray screen after stimulus (seconds)
    opts.target_fps (1,1) double = 10     % Target fps for display
    opts.resize (1,1) logical = true
    opts.verbose (1,1) logical = true
    opts.DAQ_flag (1,1) logical = false   % Enable DAQ triggering
    opts.eye (1,1) string = "right"       % Eye orientation: "none","left","right","both"
    opts.brightness (1,1) double = 1.0    % Overall luminance scale (0=black, 1=full brightness)
    opts.scope (1,1) string = '2p'        % DAQ scope: 'widefield' or '2p'
    opts.preview (1,1) logical = false    % Preview on screen only, no Teensy
end

dual_eye = strcmp(opts.eye, 'both');

% Validate and open video
assert(isfile(video_path), "Video not found: %s", video_path);
v = VideoReader(video_path);

frame_interval = 1 / opts.target_fps;

% === PRE-COMPUTE FRAMES ===
if opts.verbose
    n_vid_frames = round(v.Duration * v.FrameRate);
    if dual_eye
        fprintf('Pre-computing %d video frames for both eyes...\n', n_vid_frames);
    else
        fprintf('Pre-computing %d video frames...\n', n_vid_frames);
    end
end

% Read raw frames once (resize only)
raw_frames = {};
v.CurrentTime = 0;
while hasFrame(v)
    rgb = readFrame(v);
    if opts.resize && (size(rgb,1) ~= opts.H || size(rgb,2) ~= opts.W)
        rgb = imresize(rgb, [opts.H opts.W], 'nearest');
    end
    raw_frames{end+1} = rgb; %#ok<AGROW>
end
n_frames = length(raw_frames);

if dual_eye
    % Process separately for each eye
    frames_R   = process_frames(raw_frames, 'right', opts.brightness);
    frames_L   = process_frames(raw_frames, 'left',  opts.brightness);
    payloads_R = cellfun(@rgb565_payload, frames_R, 'UniformOutput', false);
    payloads_L = cellfun(@rgb565_payload, frames_L, 'UniformOutput', false);

    gray_img_R   = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), 'right');
    gray_img_L   = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), 'left');
    gray_payload_R = rgb565_payload(gray_img_R);
    gray_payload_L = rgb565_payload(gray_img_L);
else
    frames   = process_frames(raw_frames, opts.eye, opts.brightness);
    payloads = cellfun(@rgb565_payload, frames, 'UniformOutput', false);

    gray_frame   = rotate_for_eye(uint8(ones(opts.H, opts.W, 3) * round(127 * opts.brightness)), opts.eye);
    gray_payload = rgb565_payload(gray_frame);
end

% === BALANCED SEMI-RANDOM EYE ORDER (both mode) ===
if dual_eye
    assert(mod(opts.n_repeats, 2) == 0, ...
        'eye="both" requires an even number of repeats (got %d). Choose an even value.', opts.n_repeats);
    n_right = opts.n_repeats / 2;
    n_left  = opts.n_repeats / 2;
    rng(0);
    shuffle  = randperm(opts.n_repeats);
    eye_seq  = [ones(1, n_right) 2*ones(1, n_left)];
    eye_seq  = eye_seq(shuffle);   % 1=right, 2=left
end

% Total duration for DAQ initialization
trial_dur = opts.pre_time + v.Duration + opts.post_time;
total_dur = trial_dur * opts.n_repeats;

if opts.verbose
    fprintf('=== Teensy Trial-Based Stimulus ===\n');
    fprintf('Video: %s | %dx%d @ %.2f fps (%.2f s, %d frames)\n', ...
        video_path, v.Height, v.Width, v.FrameRate, v.Duration, n_frames);
    fprintf('Target FPS: %.2f (%.2f ms/frame)\n', opts.target_fps, 1000*frame_interval);
    fprintf('Trial structure:\n');
    fprintf('  PRE:  %.2f s (gray)\n', opts.pre_time);
    fprintf('  STIM: %.2f s (video)\n', v.Duration);
    fprintf('  POST: %.2f s (gray)\n', opts.post_time);
    fprintf('  Total/trial: %.2f s\n', trial_dur);
    fprintf('Number of repeats: %d\n', opts.n_repeats);
    fprintf('Total experiment: %.2f s (%.2f min)\n', total_dur, total_dur/60);
    if dual_eye
        fprintf('Eye mode: BOTH (semi-random per trial)\n');
        fprintf('  Right port: %s | Left port: %s\n', opts.PORT, opts.PORT_left);
        fprintf('  Eye sequence (1=R,2=L): %s\n', mat2str(eye_seq));
    else
        fprintf('Eye orientation: %s\n', opts.eye);
        if ~opts.preview
            fprintf('Port: %s\n', opts.PORT);
        end
    end
    fprintf('Brightness: %.2f\n', opts.brightness);
    fprintf('DAQ triggering: %s\n', string(opts.DAQ_flag));
    if opts.DAQ_flag
        fprintf('Scope: %s\n', opts.scope);
    end
    if opts.preview
        fprintf('MODE: PREVIEW (screen only, no Teensy)\n');
    end
    fprintf('====================================\n\n');
end

% === SETUP ===
if opts.preview
    if dual_eye
        figure('Name', 'Stimulus Preview (Both Eyes)', 'NumberTitle', 'off');
        ax_L = subplot(1,2,1); imshow(gray_img_L, 'Parent', ax_L); title(ax_L, 'Left eye');
        ax_R = subplot(1,2,2); imshow(gray_img_R, 'Parent', ax_R); title(ax_R, 'Right eye');
        ih_R = findobj(ax_R, 'Type', 'image');
        ih_L = findobj(ax_L, 'Type', 'image');
    else
        fig = figure('Name', 'Stimulus Preview', 'NumberTitle', 'off');
        ax = axes(fig);
        img_handle = imshow(gray_frame, 'Parent', ax);
        title(ax, 'Preview - Gray');
    end
    drawnow;
else
    % Initialize DAQ
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

    % Open serial port(s)
    if dual_eye
        s_right = serialport(opts.PORT,      opts.baud, "Timeout", 2);
        configureTerminator(s_right, "LF"); flush(s_right);
        s_right.OutputBufferSize = 300000;

        s_left  = serialport(opts.PORT_left, opts.baud, "Timeout", 2);
        configureTerminator(s_left,  "LF"); flush(s_left);
        s_left.OutputBufferSize = 300000;
    else
        s = serialport(opts.PORT, opts.baud, "Timeout", 2);
        configureTerminator(s, "LF");
        flush(s);
        s.OutputBufferSize = 300000;
    end
end

% === INITIALIZE LOGGING ===
stim_log = struct();
stim_log.parameters  = opts;
stim_log.trial_start = zeros(1, opts.n_repeats);
stim_log.pre_start   = zeros(1, opts.n_repeats);
stim_log.stim_start  = zeros(1, opts.n_repeats);
stim_log.stim_end    = zeros(1, opts.n_repeats);
stim_log.post_end    = zeros(1, opts.n_repeats);
stim_log.frame_times = cell(1, opts.n_repeats);
if dual_eye
    stim_log.eye_order = eye_seq;
end

if opts.verbose && ~opts.preview
    fprintf('Press Enter when ready...\n');
    pause();
end

% Start timing and trigger DAQ
if opts.DAQ_flag
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
for trial = 1:opts.n_repeats
    if opts.verbose
        if dual_eye
            eye_label = 'right'; if eye_seq(trial) == 2, eye_label = 'left'; end
            fprintf('--- Trial %d/%d [active eye: %s] ---\n', trial, opts.n_repeats, eye_label);
        else
            fprintf('--- Trial %d/%d ---\n', trial, opts.n_repeats);
        end
    end

    trial_frame_times = [];
    trial_offset = (trial - 1) * trial_dur;

    % Resolve per-trial eye assets (dual_eye only)
    if dual_eye
        if eye_seq(trial) == 1   % right eye active
            stim_payloads = payloads_R;
            stim_frames_t = frames_R;
            gray_img_idle_img = gray_img_L;
            if ~opts.preview
                s_stim = s_right;  s_idle = s_left;
                gray_payload_idle = gray_payload_L;
            end
        else                     % left eye active
            stim_payloads = payloads_L;
            stim_frames_t = frames_L;
            gray_img_idle_img = gray_img_R;
            if ~opts.preview
                s_stim = s_left;   s_idle = s_right;
                gray_payload_idle = gray_payload_R;
            end
        end
    end

    % Mark trial start
    stim_log.trial_start(trial) = toc(experiment_start);

    % === PRE PHASE (Gray screen) ===
    stim_log.pre_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  PRE start:  %.4f s\n', stim_log.pre_start(trial));
    end

    if opts.preview
        if dual_eye
            set(ih_R, 'CData', gray_img_R); title(ax_R, sprintf('Trial %d PRE', trial));
            set(ih_L, 'CData', gray_img_L); title(ax_L, sprintf('Trial %d PRE', trial));
        else
            set(img_handle, 'CData', gray_frame);
            title(ax, sprintf('Trial %d - PRE (gray)', trial));
        end
        drawnow;
    end

    frame_count = 0;
    phase_end = trial_offset + opts.pre_time;
    while toc(experiment_start) < phase_end
        if ~opts.preview
            if dual_eye
                writeline(s_right, sprintf('IMG %d %d', opts.W, opts.H));
                write(s_right, gray_payload_R, 'uint8');
                writeline(s_left,  sprintf('IMG %d %d', opts.W, opts.H));
                write(s_left,  gray_payload_L, 'uint8');
            else
                writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
                write(s, gray_payload, 'uint8');
            end
        end

        frame_count = frame_count + 1;
        trial_frame_times(end+1) = toc(experiment_start); %#ok<AGROW>

        next_frame = trial_offset + frame_count * frame_interval;
        while toc(experiment_start) < next_frame
            pause(0.0001);
        end
    end

    % === STIMULUS PHASE (Video) ===
    stim_log.stim_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  STIM start: %.4f s\n', stim_log.stim_start(trial));
    end

    stim_frame_count = 0;
    stim_offset = trial_offset + opts.pre_time;
    phase_end   = stim_offset + v.Duration;

    if opts.preview && dual_eye
        if eye_seq(trial) == 1
            title(ax_R, sprintf('Trial %d STIM (active)', trial));
            title(ax_L, sprintf('Trial %d STIM (gray)', trial));
        else
            title(ax_R, sprintf('Trial %d STIM (gray)', trial));
            title(ax_L, sprintf('Trial %d STIM (active)', trial));
        end
    end

    while toc(experiment_start) < phase_end
        frame_idx = mod(stim_frame_count, n_frames) + 1;

        if opts.preview
            if dual_eye
                if eye_seq(trial) == 1   % right eye active
                    set(ih_R, 'CData', stim_frames_t{frame_idx});
                    set(ih_L, 'CData', gray_img_idle_img);
                else                     % left eye active
                    set(ih_L, 'CData', stim_frames_t{frame_idx});
                    set(ih_R, 'CData', gray_img_idle_img);
                end
            else
                set(img_handle, 'CData', frames{frame_idx});
                title(ax, sprintf('Trial %d - STIM frame %d/%d', trial, frame_idx, n_frames));
            end
            drawnow;
        else
            if dual_eye
                writeline(s_stim, sprintf('IMG %d %d', opts.W, opts.H));
                write(s_stim, stim_payloads{frame_idx}, 'uint8');
                writeline(s_idle, sprintf('IMG %d %d', opts.W, opts.H));
                write(s_idle, gray_payload_idle, 'uint8');
            else
                writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
                write(s, payloads{frame_idx}, 'uint8');
            end
        end

        stim_frame_count = stim_frame_count + 1;
        trial_frame_times(end+1) = toc(experiment_start); %#ok<AGROW>

        target_time = stim_offset + stim_frame_count * frame_interval;
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
    post_offset = trial_offset + opts.pre_time + v.Duration;
    phase_end   = trial_offset + trial_dur;
    post_frame_count = 0;

    if opts.preview
        if dual_eye
            set(ih_R, 'CData', gray_img_R); title(ax_R, sprintf('Trial %d POST', trial));
            set(ih_L, 'CData', gray_img_L); title(ax_L, sprintf('Trial %d POST', trial));
        else
            set(img_handle, 'CData', gray_frame);
            title(ax, sprintf('Trial %d - POST (gray)', trial));
        end
        drawnow;
    end

    while toc(experiment_start) < phase_end
        if ~opts.preview
            if dual_eye
                writeline(s_right, sprintf('IMG %d %d', opts.W, opts.H));
                write(s_right, gray_payload_R, 'uint8');
                writeline(s_left,  sprintf('IMG %d %d', opts.W, opts.H));
                write(s_left,  gray_payload_L, 'uint8');
            else
                writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
                write(s, gray_payload, 'uint8');
            end
        end

        post_frame_count = post_frame_count + 1;
        trial_frame_times(end+1) = toc(experiment_start); %#ok<AGROW>

        next_frame = post_offset + post_frame_count * frame_interval;
        while toc(experiment_start) < next_frame
            pause(0.0001);
        end
    end

    stim_log.post_end(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  POST end:   %.4f s\n', stim_log.post_end(trial));
    end

    stim_log.frame_times{trial} = trial_frame_times;

    if opts.verbose
        trial_duration = stim_log.post_end(trial) - stim_log.trial_start(trial);
        fprintf('  Trial complete: %d frames, %.2f s (%.2f fps)\n\n', ...
            length(trial_frame_times), trial_duration, length(trial_frame_times)/trial_duration);
    end
end

stim_log.finish_time = toc(experiment_start);

% === CLEANUP ===
if opts.preview
    if dual_eye
        set(ih_R, 'CData', gray_img_R); title(ax_R, 'Done');
        set(ih_L, 'CData', gray_img_L); title(ax_L, 'Done');
    else
        set(img_handle, 'CData', gray_frame);
        title(ax, 'Done');
    end
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
    if dual_eye
        clear s_right s_left;
    else
        clear s;
    end
end

% Calculate statistics
total_frames = sum(cellfun(@length, stim_log.frame_times));
overall_fps  = total_frames / stim_log.finish_time;

if opts.verbose
    fprintf('\n=== Experiment Complete ===\n');
    fprintf('Total time: %.3f s (%.2f min)\n', stim_log.finish_time, stim_log.finish_time/60);
    fprintf('Total frames: %d\n', total_frames);
    fprintf('Overall FPS: %.2f\n', overall_fps);
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
        frame = flip(frame, 2);    % invert x axis
        frame = rot90(frame, 1);  % -90° CW
    elseif strcmp(eye, 'left')
        frame = rot90(frame, -1);   % +90° CCW
    end
end

function payload = rgb565_payload(img)
    if ~isa(img,'uint8'), img = im2uint8(img); end
    R5 = bitshift(uint16(img(:,:,1)), -3);
    G6 = bitshift(uint16(img(:,:,2)), -2);
    B5 = bitshift(uint16(img(:,:,3)), -3);
    rgb565 = bitor(bitshift(R5,11), bitor(bitshift(G6,5), B5));
    payload = typecast(reshape(rgb565.', [], 1), 'uint8');
end
