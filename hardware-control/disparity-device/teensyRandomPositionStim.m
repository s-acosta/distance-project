function Stimdata = teensyRandomPositionStim(opts)
% Play radial checkerboard ball at random x positions on Teensy display.
%
% The ball appears at each position from a predefined set, in randomized
% order, with gray blanks between presentations. Each position is shown
% n_repeats times (shuffled differently each repeat cycle).
%
% Usage:
%   teensyRandomPositionStim()
%   teensyRandomPositionStim('positions', [60 120 180], 'n_repeats', 5)
%   teensyRandomPositionStim('preview', true)  % Preview on screen only
%
% Saves Stimdata.mat via uisave with fields:
%   Stimdata.blank_on       (n_repeats x n_positions) - Blank start times (s)
%   Stimdata.blank_end      (n_repeats x n_positions) - Blank end times (s)
%   Stimdata.stim_on        (n_repeats x n_positions) - Stim start times (s)
%   Stimdata.stim_end       (n_repeats x n_positions) - Stim end times (s)
%   Stimdata.position_order (n_repeats x n_positions) - Position index shown
%   Stimdata.positions      (1 x n_positions)         - X positions (pixels)
%   Stimdata.on_time        scalar                    - Stim duration (s)
%   Stimdata.off_time       scalar                    - Blank duration (s)
%   Stimdata.repeats        scalar                    - Repeats per position

arguments
    opts.positions (1,:) double = linspace(20, 220, 10)  % x positions (pixels)
    opts.stim_size (1,1) double = 30                      % ball diameter (pixels)
    opts.rcycles (1,1) double = 1                          % radial checkerboard cycles
    opts.tcycles (1,1) double = 2                          % angular checkerboard cycles
    opts.PORT (1,1) string = "COM5"
    opts.baud (1,1) double = 115200
    opts.H (1,1) double = 240
    opts.W (1,1) double = 240
    opts.n_repeats (1,1) double = 10                       % repeats per position
    opts.on_time (1,1) double = 2.0                        % stim display duration (s)
    opts.off_time (1,1) double = 2.0                       % gray blank duration (s)
    opts.target_fps (1,1) double = 10
    opts.DAQ_flag (1,1) logical = false
    opts.verbose (1,1) logical = true
    opts.preview (1,1) logical = false
end

n_positions = length(opts.positions);
frame_interval = 1 / opts.target_fps;

% === PRE-COMPUTE FRAMES ===
if opts.verbose
    fprintf('Pre-computing %d ball frames...\n', n_positions);
end

gray_frame = uint8(ones(opts.H, opts.W, 3) * 127);
gray_payload = rgb565_payload(gray_frame);

% Pre-compute ball frame and payload for each position
ball_frames = cell(1, n_positions);
ball_payloads = cell(1, n_positions);
for i = 1:n_positions
    ball_frames{i} = make_ball_frame(opts.positions(i), opts.stim_size, ...
        opts.W, opts.H, opts.rcycles, opts.tcycles);
    ball_payloads{i} = rgb565_payload(ball_frames{i});
end

% Calculate timing
stim_dur = opts.on_time + opts.off_time;
rep_dur = stim_dur * n_positions;
total_dur = rep_dur * opts.n_repeats;

if opts.verbose
    fprintf('=== Teensy Random Position Stimulus ===\n');
    fprintf('Positions: %s\n', mat2str(round(opts.positions)));
    fprintf('Ball size: %d pixels\n', opts.stim_size);
    fprintf('On time:  %.2f s\n', opts.on_time);
    fprintf('Off time: %.2f s\n', opts.off_time);
    fprintf('Repeats per position: %d\n', opts.n_repeats);
    fprintf('Total presentations: %d\n', opts.n_repeats * n_positions);
    fprintf('Total duration: %.2f s (%.2f min)\n', total_dur, total_dur/60);
    fprintf('Target FPS: %.2f\n', opts.target_fps);
    if opts.preview
        fprintf('MODE: PREVIEW (screen only, no Teensy)\n');
    else
        fprintf('Port: %s\n', opts.PORT);
        fprintf('DAQ triggering: %s\n', string(opts.DAQ_flag));
    end
    fprintf('========================================\n\n');
end

% === SETUP ===
if opts.preview
    fig = figure('Name', 'Random Position Stimulus Preview', ...
        'NumberTitle', 'off');
    ax = axes(fig);
    img_handle = imshow(gray_frame, 'Parent', ax);
    title(ax, 'Preview - Gray');
    drawnow;
else
    % Initialize DAQ triggering
    wf = WidefieldTriggerer(opts.DAQ_flag);
    wf.initialize(total_dur);

    % Open serial
    s = serialport(opts.PORT, opts.baud, "Timeout", 2);
    configureTerminator(s, "LF");
    flush(s);
    s.OutputBufferSize = 300000;
end

% === INITIALIZE LOGGING ===
blank_on  = zeros(opts.n_repeats, n_positions);
blank_end = zeros(opts.n_repeats, n_positions);
stim_on   = zeros(opts.n_repeats, n_positions);
stim_end  = zeros(opts.n_repeats, n_positions);
position_order = zeros(opts.n_repeats, n_positions);

% Pre-compute all shuffled orders
for rep = 1:opts.n_repeats
    position_order(rep, :) = randperm(n_positions);
end

if opts.verbose && ~opts.preview
    fprintf('Press Enter when ready...\n');
    pause();
end

% Start timing and trigger DAQ
experiment_start = tic;

if ~opts.preview
    wf.start();
end

if opts.verbose
    fprintf('Starting experiment...\n\n');
end

% === MAIN LOOP ===
for rep = 1:opts.n_repeats
    order = position_order(rep, :);

    for trial = 1:n_positions
        pos_idx = order(trial);

        if opts.verbose
            fprintf('--- Rep %d/%d, Position %d (x=%d, trial %d/%d) ---\n', ...
                rep, opts.n_repeats, pos_idx, round(opts.positions(pos_idx)), ...
                trial, n_positions);
        end

        % Absolute timing targets for this trial
        trial_offset = ((rep-1) * n_positions + (trial-1)) * stim_dur;
        blank_end_target = trial_offset + opts.off_time;
        stim_end_target  = trial_offset + stim_dur;

        % === BLANK PHASE (Gray screen) ===
        blank_on(rep, trial) = toc(experiment_start);
        if opts.verbose
            fprintf('  Blank start: %.4f s\n', blank_on(rep, trial));
        end

        if opts.preview
            set(img_handle, 'CData', gray_frame);
            title(ax, 'Gray blank');
            drawnow;
        end

        frame_count = 0;
        while toc(experiment_start) < blank_end_target
            if ~opts.preview
                writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
                write(s, gray_payload, 'uint8');
            end

            frame_count = frame_count + 1;
            next_frame = trial_offset + frame_count * frame_interval;
            while toc(experiment_start) < next_frame
                pause(0.0001);
            end
        end

        blank_end(rep, trial) = toc(experiment_start);

        % === STIM PHASE (Ball at position) ===
        stim_on(rep, trial) = toc(experiment_start);
        if opts.verbose
            fprintf('  Stim start:  %.4f s (x=%d)\n', stim_on(rep, trial), ...
                round(opts.positions(pos_idx)));
        end

        if opts.preview
            set(img_handle, 'CData', ball_frames{pos_idx});
            title(ax, sprintf('Position %d (x=%d)', pos_idx, round(opts.positions(pos_idx))));
            drawnow;
        end

        stim_frame_count = 0;
        while toc(experiment_start) < stim_end_target
            if ~opts.preview
                writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
                write(s, ball_payloads{pos_idx}, 'uint8');
            end

            stim_frame_count = stim_frame_count + 1;
            target_time = trial_offset + opts.off_time + stim_frame_count * frame_interval;
            while toc(experiment_start) < target_time
                pause(0.0001);
            end
        end

        stim_end(rep, trial) = toc(experiment_start);
        if opts.verbose
            fprintf('  Stim end:    %.4f s (duration: %.4f s)\n', ...
                stim_end(rep, trial), stim_end(rep, trial) - stim_on(rep, trial));
        end
    end

    if opts.verbose
        fprintf('\n');
    end
end

finish_time = toc(experiment_start);

% === CLEANUP ===
if opts.preview
    set(img_handle, 'CData', gray_frame);
    title(ax, 'Done');
    drawnow;
else
    wf.finish();
    clear s;
end

if opts.verbose
    fprintf('\n=== Experiment Complete ===\n');
    fprintf('Total time: %.3f s (%.2f min)\n', finish_time, finish_time/60);
    fprintf('Total presentations: %d\n', opts.n_repeats * n_positions);
end

% Build Stimdata struct
Stimdata.blank_on       = blank_on;
Stimdata.blank_end      = blank_end;
Stimdata.stim_on        = stim_on;
Stimdata.stim_end       = stim_end;
Stimdata.position_order = position_order;
Stimdata.positions      = opts.positions;
Stimdata.off_time       = opts.off_time;
Stimdata.on_time        = opts.on_time;
Stimdata.repeats        = opts.n_repeats;
Stimdata.stim_size      = opts.stim_size;

% % Save
% uisave('Stimdata');

end

%% Helper functions
function frame = make_ball_frame(x_pos, stim_size, W, H, rcycles, tcycles)
% Build radial checkerboard ball at position (x_pos, H/2) on gray background.
% Math reused from generate_leftright_stim.m lines 62-69.

    xylim = pi * rcycles;
    [xg, yg] = meshgrid(linspace(-xylim, xylim, stim_size));
    at = atan2(yg, xg);
    checksN = (1 + sign(sin(at * tcycles) + eps) .* sign(sin(sqrt(xg.^2 + yg.^2)))) / 2;
    circle = (xg.^2 + yg.^2) <= xylim^2;
    ball = circle .* checksN + 0.5 * ~circle;
    ball_rgb = uint8(round(ball * 255));
    ball_rgb = cat(3, ball_rgb, ball_rgb, ball_rgb);

    % Gray background
    frame = uint8(ones(H, W, 3) * 127);
    y_center = round(H / 2);
    half = floor(stim_size / 2);

    % Placement bounds (clipped to frame edges)
    y1 = max(1, y_center - half);
    y2 = min(H, y_center - half + stim_size - 1);
    x1 = max(1, round(x_pos) - half);
    x2 = min(W, round(x_pos) - half + stim_size - 1);

    % Corresponding region in the ball texture
    by1 = y1 - (y_center - half) + 1;
    by2 = by1 + (y2 - y1);
    bx1 = x1 - (round(x_pos) - half) + 1;
    bx2 = bx1 + (x2 - x1);

    frame(y1:y2, x1:x2, :) = ball_rgb(by1:by2, bx1:bx2, :);
end

function payload = rgb565_payload(img)
    if ~isa(img,'uint8'), img = im2uint8(img); end
    R5 = bitshift(uint16(img(:,:,1)), -3);
    G6 = bitshift(uint16(img(:,:,2)), -2);
    B5 = bitshift(uint16(img(:,:,3)), -3);
    rgb565 = bitor(bitshift(R5,11), bitor(bitshift(G6,5), B5));
    payload = typecast(reshape(rgb565.', [], 1), 'uint8');
end
