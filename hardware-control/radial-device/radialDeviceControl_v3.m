function stim_data = radialDeviceControl_v3(motor_port, options)
% Radial Device Control v3 - left / right / both eyes in interleaved blocks, sweeps in both directions
%
% Usage:
%   plan      = radialDeviceControl_v3([], 'n_repeats', 6, 'preview', true)   % plan and timing only
%   stim_data = radialDeviceControl_v3('COM3', 'trigger_port', 'COM5', 'n_repeats', 6)   % starts on the scope
%   stim_data = radialDeviceControl_v3('COM3', 'trigger', "daq", 'n_repeats', 6)         % old rig, NI DAQ
%
% Design:
%   Eye conditions run in blocks, in the order of block_order. The default, L R B B R L, puts each
%   condition's two blocks symmetrically in time, so slow drift (bleaching, arousal) averages out
%   across conditions. The blocker is moved by hand: between blocks the script announces the next
%   condition and waits switch_time seconds, so the session length is known in advance.
%   Each trial is one sweep with its own off period (ball raised, swept, held, lowered). The arm
%   doesn't return between trials, so sweeps alternate start -> end (forward) and end -> start
%   (reverse), both at forward_speed. Within each block every distance gets the same number of
%   forward and reverse sweeps, in random order.
%
% Options:
%   distances      - Distance ids (= servo ids); 4 is the control (no ball raised). Default: [0 1 2 3 4]
%   n_repeats      - Sweeps per distance, per eye condition, per direction, over the whole session
%                    (split evenly across that condition's blocks). Default: 6
%   block_order    - Eye condition of each block: "left", "right" or "both".
%                    Default: ["left" "right" "both" "both" "right" "left"]
%   switch_time    - Seconds allowed to move the blocker between blocks. Default: 30
%   start_angle    - Where forward sweeps start and reverse sweeps end (deg). Default: -30
%   end_angle      - Where forward sweeps end and reverse sweeps start (deg). Default: 60
%   off_time       - Seconds from trial start to sweep start; the ball is raised inside it. Default: 1
%   raise_time     - ms to raise the servo. Default: 250
%   lower_time     - ms to lower the servo. Default: 100
%   forward_speed  - Stepper step delay (us) for every sweep, both directions. Default: 28500
%   return_speed   - Step delay (us) for the initial positioning and the final return. Default: 10000
%   time_per_trial - Seconds per trial; [] = the shortest that fits one sweep, rounded up to 0.5 s.
%   frame_rate     - Imaging rate (Hz), only for the frame estimate. Default: 10
%   random_flag    - Shuffle the sweep order within blocks. Default: true
%   trigger        - How the session starts. Default: "arduino"
%                    "arduino": on the scope's first frame. The start-trigger Arduino (startTrigger.ino)
%                               on trigger_port reads the frame clock; start the acquisition on the scope.
%                    "daq":     on enter; the NI DAQ (Dev1/ao0) goes to 5 V to start the scope (old rig).
%                    "enter":   on enter, with no trigger (dry runs).
%   trigger_port   - COM port of the start-trigger Arduino, needed for "arduino". Default: ""
%   preview        - Print the plan and timing and return without touching the hardware. Default: false
%
% Eye condition codes (eye_sequence and block_order in stim_data): 1 = left eye only (blocker on the
% right eye), 2 = right eye only (blocker on the left eye), 3 = both eyes (no blocker).
%
% Saved per trial: distance_sequence, eye_sequence, direction_sequence (+1 start -> end,
% -1 end -> start), sweep_from / sweep_to (deg), block_sequence, off_start, on_start (sweep start),
% on_end (sweep end). Per block: block_start. Per switch: switch_start, switch_end. Times are seconds
% from the session start (the scope's first frame with trigger "arduino").
% monocular_sequence (true when one eye is covered) is kept so the existing loaders read the file.
% Every field is numeric or logical, so scipy.io.loadmat reads it.

arguments
    motor_port
    options.distances = [0, 1, 2, 3, 4];
    options.n_repeats = 6;
    options.block_order = ["left", "right", "both", "both", "right", "left"];
    options.switch_time = 30;  % seconds
    options.start_angle = -30;
    options.end_angle = 60;
    options.off_time = 1;  % seconds
    options.raise_time = 250;  % ms for raising servo
    options.lower_time = 100;  % ms for lowering servo
    options.forward_speed = 28500;  % us delay, every sweep
    options.return_speed = 10000;   % us delay, positioning only
    options.time_per_trial = [];
    options.frame_rate = 10;  % Hz
    options.random_flag = true;
    options.trigger (1,1) string {mustBeMember(options.trigger, ["arduino", "daq", "enter"])} = "arduino";
    options.trigger_port = "";
    options.preview = false;
end

CONTROL = 4;  % distance id with no ball raised, as in v2
EYE_LABELS = ["left", "right", "both"];
EYE_PROMPT = ["LEFT EYE ONLY  - blocker on the RIGHT eye", ...
              "RIGHT EYE ONLY - blocker on the LEFT eye", ...
              "BOTH EYES      - remove the blocker"];

%% Block design
block_order = string(options.block_order);
[is_eye, block_eye] = ismember(block_order, EYE_LABELS);
if ~all(is_eye)
    error('block_order entries must be "left", "right" or "both"; got: %s', strjoin(block_order(~is_eye), ', '));
end
blocks_per_eye = accumarray(block_eye(:), 1, [3 1])';
used = blocks_per_eye > 0;
if numel(unique(blocks_per_eye(used))) > 1
    error('Every eye condition in block_order needs the same number of blocks (now %s for left/right/both).', ...
          mat2str(blocks_per_eye));
end
k = max(blocks_per_eye);  % blocks per eye condition
if mod(options.n_repeats, k) ~= 0
    error('n_repeats (%d) must be divisible by the number of blocks per eye condition (%d).', options.n_repeats, k);
end
per_block = options.n_repeats / k;  % sweeps per distance and direction in each block

distances = options.distances(:)';
n_distances = numel(distances);
n_blocks = numel(block_order);
n_cond = nnz(used);
sweeps_per_block = 2 * n_distances * per_block;

% Within a block, forward and reverse sweeps alternate. Blocks have an even number of sweeps,
% so the arm ends each block where it started and every block starts with a forward sweep.
n_trials = n_blocks * sweeps_per_block;
direction_vec = repmat([1, -1], 1, n_trials / 2);
eye_vec = repelem(block_eye(:)', sweeps_per_block);
block_vec = repelem(1:n_blocks, sweeps_per_block);
distance_vec = zeros(1, n_trials);
for b = 1:n_blocks
    fwd = repmat(distances, 1, per_block);
    rev = repmat(distances, 1, per_block);
    if options.random_flag
        fwd = fwd(randperm(numel(fwd)));
        rev = rev(randperm(numel(rev)));
    end
    idx = (b - 1) * sweeps_per_block + (1:sweeps_per_block);
    distance_vec(idx(1:2:end)) = fwd;
    distance_vec(idx(2:2:end)) = rev;
end
sweep_from = options.start_angle * (direction_vec == 1) + options.end_angle * (direction_vec == -1);
sweep_to = options.end_angle * (direction_vec == 1) + options.start_angle * (direction_vec == -1);

%% Timing
% Movement time scales with angle x per-step delay: 60 deg at 28500 us took 4.03-4.08 s, so 4.1 s
% is used (as in v2). Servo moves add ~0.1 s of command pause + reply, and the raise happens
% inside the off period. A trial is: off period, sweep, 0.5 s hold, lower.
sweep_s = abs(options.end_angle - options.start_angle) * options.forward_speed * 4.1 / (60 * 28500);
off_period = max(options.off_time, options.raise_time / 1000 + 0.1);
lower_s = options.lower_time / 1000 + 0.1;
trial_content = off_period + sweep_s + 0.5 + lower_s;
min_trial = ceil(2 * (trial_content + 0.3)) / 2;  % 0.3 s margin, rounded up to 0.5 s
if isempty(options.time_per_trial)
    options.time_per_trial = min_trial;
elseif options.time_per_trial < min_trial
    fprintf('time_per_trial raised from %g to %g s (one trial takes ~%.1f s)\n', ...
            options.time_per_trial, min_trial, trial_content);
    options.time_per_trial = min_trial;
end
% A switch window only where the eye condition changes (none between the two "both" blocks)
switch_before = [false, block_eye(2:end) ~= block_eye(1:end - 1)];
n_switches = nnz(switch_before);
session_s = @(n) n * options.time_per_trial + n_switches * options.switch_time;
total_s = session_s(n_trials);
mmss = @(sec) sprintf('%2d:%02d', floor(round(sec) / 60), mod(round(sec), 60));

fprintf('========================================\n');
fprintf('Session plan\n');
fprintf('  Blocks: %s\n', strjoin(block_order, ' -> '));
fprintf('  Distances: %s (%d = control, no ball)\n', mat2str(distances), CONTROL);
fprintf('  Per distance and eye condition: %d forward + %d reverse sweeps (%d + %d in each block)\n', ...
        options.n_repeats, options.n_repeats, per_block, per_block);
fprintf('  Sweeps per block: %d   Blocks: %d   Total sweeps: %d\n', sweeps_per_block, n_blocks, n_trials);
fprintf('  Sweep: %d <-> %d deg, %.2f s at %d us per step\n', ...
        options.start_angle, options.end_angle, sweep_s, options.forward_speed);
fprintf('  Trial: %.1f s (off %.2f + sweep %.2f + hold 0.50 + lower %.2f = %.2f s)\n', ...
        options.time_per_trial, off_period, sweep_s, lower_s, trial_content);
fprintf('  Blocker switches: %d x %g s\n', n_switches, options.switch_time);
fprintf('  Session: %.0f s (%.1f min), ~%.0f frames at %g Hz\n', ...
        total_s, total_s / 60, total_s * options.frame_rate, options.frame_rate);
fprintf('  Schedule (approx, mm:ss from the session start):\n');
fprintf('     0:00  before the start: %s\n', EYE_PROMPT(block_eye(1)));
for b = 1:n_blocks
    t0 = (b - 1) * sweeps_per_block * options.time_per_trial + nnz(switch_before(1:b)) * options.switch_time;
    if switch_before(b)
        fprintf('    %s  SWITCH (%g s): %s\n', mmss(t0 - options.switch_time), options.switch_time, ...
                EYE_PROMPT(block_eye(b)));
    end
    fprintf('    %s  block %d (%s), trials %d-%d\n', mmss(t0), b, block_order(b), ...
            (b - 1) * sweeps_per_block + 1, b * sweeps_per_block);
end
fprintf('    %s  end\n', mmss(total_s));

% Plan saved in every mode; the run adds the timestamps
stim_data = rmfield(options, 'preview');
stim_data.trigger = char(options.trigger);  % char: scipy can't read string objects
stim_data.trigger_port = char(options.trigger_port);
stim_data.block_order = block_eye(:)';
stim_data.distance_sequence = distance_vec;
stim_data.eye_sequence = eye_vec;
stim_data.direction_sequence = direction_vec;
stim_data.sweep_from = sweep_from;
stim_data.sweep_to = sweep_to;
stim_data.block_sequence = block_vec;
stim_data.monocular_sequence = eye_vec ~= 3;

if options.preview
    fprintf('\n  Session length vs n_repeats (sweeps per distance, eye condition and direction):\n');
    fprintf('    n_repeats   sweeps   minutes\n');
    for r = unique([k:k:8 * k, options.n_repeats])
        n = n_cond * n_distances * 2 * r;
        fprintf('    %9d   %6d   %7.1f\n', r, n, session_s(n) / 60);
    end
    fprintf('========================================\n');
    return
end
fprintf('========================================\n');

if options.trigger == "arduino" && strlength(options.trigger_port) == 0
    error('trigger "arduino" needs trigger_port, the start-trigger Arduino''s COM port (e.g. ''COM5'').');
end

%% Initialize DAQ
if options.trigger == "daq"
    DAQ = daq('ni');
    DAQ.addoutput('Dev1', 'ao0', 'Voltage');
    DAQ.write(0);
    fprintf('DAQ initialized\n');
end

% Connect to Arduino
fprintf('Connecting to Arduino on %s...\n', motor_port);
motor = serialport(motor_port, 115200);
configureTerminator(motor, "LF");
flush(motor);

% Wait for Arduino ready
while true
    if motor.NumBytesAvailable > 0
        line = readline(motor);
        fprintf('  %s\n', line);
        if contains(line, "Arduino is ready")
            fprintf('Connected!\n\n');
            break;
        end
    end
    pause(0.1);
end

% Clear any initial data
pause(0.2);
flush(motor);

% Connect to the start-trigger Arduino (opening the port resets it)
if options.trigger == "arduino"
    fprintf('Connecting to the start-trigger Arduino on %s...\n', options.trigger_port);
    frame_trigger = serialport(options.trigger_port, 115200);
    configureTerminator(frame_trigger, "LF");
    if ~waitForLine(frame_trigger, "Trigger ready", 10)
        error('No "Trigger ready" from %s: is startTrigger.ino loaded?', options.trigger_port);
    end
    fprintf('Connected!\n\n');
end

% Initial positioning: the first sweep is forward
fprintf('Moving to starting position %d deg...\n', sweep_from(1));
sendCommand(motor, sprintf("move %d %d", sweep_from(1), options.return_speed));
waitForMovementDone(motor);
current_angle = sweep_from(1);
fprintf('Ready at %d deg\n\n', current_angle);

fprintf('Set the blocker for block 1: %s\n', EYE_PROMPT(eye_vec(1)));

% Data collection
off_start = zeros(1, n_trials);
on_start = zeros(1, n_trials);
on_end = zeros(1, n_trials);
block_start = zeros(1, n_blocks);
switch_start = zeros(1, n_switches);
switch_end = zeros(1, n_switches);
s = 0;

% Start timing: on the scope's first frame ("arduino"), or on enter ("daq" also starts the scope)
switch options.trigger
    case "arduino"
        fprintf('Stop focus / live view on the scope, then press enter to arm the trigger\n');
        pause();
        flush(frame_trigger);
        writeline(frame_trigger, "a");
        if ~waitForLine(frame_trigger, "Armed", 5)
            error('No "Armed" reply from the start-trigger Arduino on %s', options.trigger_port);
        end
        fprintf('Armed: start the acquisition on the scope. The session starts on the first frame.\n');
        waitForLine(frame_trigger, "Start", inf);  % Ctrl-C aborts
        tic;
        fprintf('First frame received: session started\n\n');
    case "daq"
        fprintf('Press enter when ready (timing and the DAQ trigger start then)\n');
        pause();
        tic;
        DAQ.write(5);
        fprintf('DAQ trigger sent (5V)\n\n');
    case "enter"
        fprintf('Press enter when ready (timing starts then; no trigger)\n');
        pause();
        tic;
end

% Main experiment loop
for trial = 1:n_trials
    b = block_vec(trial);

    % Blocker switch window before a block whose eye condition differs from the last one
    if trial > 1 && b ~= block_vec(trial - 1) && ~switch_before(b)
        fprintf('\n  Block %d (%s) starts: same eye condition, no switch\n\n', b, block_order(b));
    end
    if trial > 1 && b ~= block_vec(trial - 1) && switch_before(b)
        s = s + 1;
        switch_start(s) = toc;
        fprintf('\n****************************************\n');
        fprintf('  SWITCH BLOCKER: %s\n', EYE_PROMPT(eye_vec(trial)));
        fprintf('  %g s, then block %d of %d starts\n', options.switch_time, b, n_blocks);
        fprintf('****************************************\n');
        beep;
        last = inf;
        while toc < switch_start(s) + options.switch_time
            remaining = ceil(switch_start(s) + options.switch_time - toc);
            if mod(remaining, 5) == 0 && remaining < last
                fprintf('    %d s left\n', remaining);
                last = remaining;
                if remaining == 5
                    beep;
                end
            end
            pause(0.01);
        end
        switch_end(s) = toc;
    end

    % Sweeps alternate, so the arm should already be at this sweep's start
    if current_angle ~= sweep_from(trial)
        warning('Arm at %d deg, expected %d deg: repositioning (this trial runs long)', ...
                current_angle, sweep_from(trial));
        sendCommand(motor, sprintf("move %d %d", sweep_from(trial), options.return_speed));
        waitForMovementDone(motor);
    end

    off_start(trial) = toc;
    if trial == 1 || b ~= block_vec(trial - 1)
        block_start(b) = off_start(trial);
    end
    if direction_vec(trial) > 0
        direction = "forward";
    else
        direction = "reverse";
    end
    fprintf('  Trial %d/%d | block %d (%s) | distance %d | %s %d -> %d\n', trial, n_trials, b, ...
            EYE_LABELS(eye_vec(trial)), distance_vec(trial), direction, sweep_from(trial), sweep_to(trial));

    % Raise servo
    if distance_vec(trial) ~= CONTROL
        sendCommand(motor, sprintf("raise %d %d", distance_vec(trial), options.raise_time));
        waitForServoDone(motor);
    end

    % Wait for off_time
    while toc < off_start(trial) + options.off_time
        pause(0.01);
    end

    % Sweep
    on_start(trial) = toc;
    sendCommand(motor, sprintf("move %d %d", sweep_to(trial), options.forward_speed));
    waitForMovementDone(motor);
    on_end(trial) = toc;
    current_angle = sweep_to(trial);
    fprintf('    off %.2f  on %.2f - %.2f s\n', off_start(trial), on_start(trial), on_end(trial));

    % Stay for half a second, then lower
    while toc < on_end(trial) + 0.5
        pause(0.01);
    end
    if distance_vec(trial) ~= CONTROL
        sendCommand(motor, sprintf("lower %d %d", distance_vec(trial), options.lower_time));
        waitForServoDone(motor);
    end

    while toc < off_start(trial) + options.time_per_trial
        pause(0.01);
    end
end

% Turn off DAQ trigger
if options.trigger == "daq"
    DAQ.write(0);
    fprintf('DAQ trigger ended (0V)\n');
end

% Return to zero position after experiment
fprintf('Experiment complete - returning to 0 deg...\n');
sendCommand(motor, sprintf("move 0 %d", options.return_speed));
waitForMovementDone(motor);
fprintf('At rest position (0 deg)\n\n');
fprintf('Total time: %.2f seconds (planned %.0f)\n', toc, total_s);

% Package timestamps
stim_data.off_start = off_start;
stim_data.on_start = on_start;
stim_data.on_end = on_end;
stim_data.block_start = block_start;
stim_data.switch_start = switch_start;
stim_data.switch_end = switch_end;

% Close connections
clear motor frame_trigger;

%% Helper Functions

    function found = waitForLine(port, pattern, timeout)
        % Wait for a line containing pattern; false on timeout (inf waits until Ctrl-C)
        start_time = tic;
        while toc(start_time) < timeout
            if port.NumBytesAvailable > 0
                if contains(readline(port), pattern)
                    found = true;
                    return;
                end
            end
            pause(0.001);
        end
        found = false;
    end

    function sendCommand(port, cmd)
        % Send command and wait for acknowledgment
        writeline(port, cmd);
        pause(0.05);  % Brief pause for command processing

        % Read and discard the acknowledgment
        if port.NumBytesAvailable > 0
            readline(port);
        end
    end

    function waitForMovementDone(port)
        % Wait for "<Movement done: XX>" message
        timeout = 30;
        start_time = tic;

        while toc(start_time) < timeout
            if port.NumBytesAvailable > 0
                line = readline(port);

                if contains(line, "Movement done")
                    return;
                end
            end
            pause(0.01);
        end

        warning('Timeout waiting for movement completion');
    end

    function waitForServoDone(port)
        % Wait for "<Servo done>" message
        timeout = 10;
        start_time = tic;

        while toc(start_time) < timeout
            if port.NumBytesAvailable > 0
                line = readline(port);

                if contains(line, "Servo done")
                    return;
                end
            end
            pause(0.01);
        end

        warning('Timeout waiting for servo completion');
    end

end
