function stim_data = radialDeviceControl_v2(motor_port, options)
% Radial Device Control - Matlab Interface
%
% Usage:
%   stim_data = radialDeviceControl_v2('COM3')
%   stim_data = radialDeviceControl_v2('COM3', 'servo_id', 2, 'n_repeats', 5)
%
% Options:
%   servo_id       - Which servo to use (0-3), default: 0
%   n_repeats      - Number of trial repetitions, default: 2
%   start_angle    - Starting angle in degrees, default: -30
%   end_angle      - Ending angle in degrees, default: 60
%   off_time       - Wait time before servo raise in seconds, default: 1
%   raise_time     - Time to raise servo in ms, default: 250
%   lower_time     - Time to lower servo in ms, default: 100
%   forward_speed  - Speed for start->end movement (us), default: 27000
%   return_speed   - Speed for end->start movement (us), default: 10000
%   monocular      - Enable monocular sequence blocks, default: false
%   DAQ_flag       - Enable DAQ triggering, default: false

arguments
    motor_port
    options.distances = [0, 1, 2, 3, 4];
    options.time_per_trial = 12;
    options.n_repeats = 2;
    options.start_angle = -30;
    options.end_angle = 60;
    options.off_time = 1;  % seconds
    options.raise_time = 250;  % ms for raising servo
    options.lower_time = 100;  % ms for lowering servo
    options.forward_speed = 28500;  % us delay for -30 to 60
    options.return_speed = 10000;   % us delay for 60 to -30
    options.monocular = false;
    options.random_flag = true;
    options.DAQ_flag = false;
end

%% Initialize DAQ
if options.DAQ_flag
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

%% Generate stimulus structure
n_distances = length(options.distances);

if options.random_flag
    distance_vec = [];
    for i = 1:options.n_repeats
        distance_vec = [distance_vec options.distances(randperm(n_distances))];
    end

else
    distance_vec = repmat(options.distances, [1 options.n_repeats]);
end

if options.monocular
    % Ensure the total number of trials is a multiple of 8
    total_trials = options.n_repeats * n_distances;
    if mod(options.n_repeats, 8) ~= 0
        error(['For proper monocular sequence blocks, total trials must be a multiple of 8. ' ...
            'Current total_trials = %d. Please adjust n_repeats or distances.'], total_trials);
    end
    % Create sequence blocks
    block_size = total_trials / 8;
    pattern = [true(1, block_size), false(1, block_size)];
    repeats = total_trials / (2 * block_size);  % Should be 4 for 8 blocks
    monocular_sequence = repmat(pattern, 1, repeats);
else
    monocular_sequence = false(1, options.n_repeats * n_distances);
end

% Initialize data collection
off_start = zeros(1, options.n_repeats);
on_start = zeros(1, options.n_repeats);
on_end = zeros(1, options.n_repeats);


% If monocular mode, inform user when to switch between monocular and binocular
if options.monocular
    fprintf('Monocular mode: ENABLED\n');
    fprintf('Monocular sequence switches:\n');
    
    % Find where the sequence changes (transitions between true/false)
    transitions = find(diff([false, monocular_sequence]) ~= 0);
    
    % Display the transitions in a readable format
    for i = 1:length(transitions)
        trial_num = transitions(i);
        is_monocular = monocular_sequence(trial_num);
        
        if is_monocular
            fprintf('  Trial %3d: Switch to MONOCULAR\n', trial_num);
        else
            fprintf('  Trial %3d: Switch to BINOCULAR\n', trial_num);
        end
    end
    
    fprintf('Monocular trials: %d, Binocular trials: %d\n', sum(monocular_sequence), sum(~monocular_sequence));
else
    fprintf('Monocular mode: DISABLED (all binocular)\n');
end

fprintf('========================================\n');
fprintf('Press Ctrl+C to stop\n\n');

% Initial positioning: Move to start position
fprintf('Initial Setup\n');
fprintf('  Moving to starting position %d°...\n', options.start_angle);
sendCommand(motor, sprintf("move %d %d", options.start_angle, options.return_speed));
waitForMovementDone(motor);
fprintf('  Ready at %d°\n\n', options.start_angle);

fprintf('========================================\n');
fprintf('Experiment Configuration\n');
fprintf('Number of repeats: %d\n', options.n_repeats);
fprintf('Number of total repeats: %d\n', length(distance_vec));
fprintf('Angles: %d° -> %d°\n', options.start_angle, options.end_angle);
fprintf('Off time: %.2f seconds\n', options.off_time);
fprintf('Servo timing: raise=%dms, lower=%dms\n', options.raise_time, options.lower_time);
fprintf('Stepper speeds: forward=%dus, return=%dus\n', options.forward_speed, options.return_speed);
fprintf('DAQ triggering: %s\n', options.DAQ_flag);

estimated_total_time = options.time_per_trial * length(distance_vec);
estimated_frames = estimated_total_time / 0.1;

fprintf('\nEstimated experiment duration:\n');
fprintf('  Time: %.2f seconds (%.2f minutes)\n', estimated_total_time, estimated_total_time/60);
fprintf('  Frames: %.0f (100ms/frame)\n', estimated_frames);
fprintf('========================================\n');

fprintf('Press enter when ready\n');
pause();

% Start timing and trigger DAQ
tic;

if options.DAQ_flag
    DAQ.write(5);
    fprintf('DAQ trigger sent (5V)\n\n');
end

% Main experiment loop
for trial = 1:length(distance_vec)

    % Record off_start time
    off_start(trial) = toc;
    fprintf('  Off start: %.4fs\n', off_start(trial));
    
    fprintf('\tPresentation %d | Distance %d\n', trial, distance_vec(trial));
    if options.monocular
        if monocular_sequence(trial)
            fprintf(' \tMonocular Mode\n');
        else
            fprintf('\tBinocular Mode\n');
        end
    else
        fprintf('\n');
    end
    
    % Raise servo
    if distance_vec(trial) ~= 4
        sendCommand(motor, sprintf("raise %d %d", distance_vec(trial), options.raise_time));
        waitForServoDone(motor);
    end
    
    % Wait for off_time
    while toc < off_start(trial) + options.off_time
        pause(0.01);
    end
    
    % Record on_start time and begin movement
    on_start(trial) = toc;
    fprintf('  On start: %.4fs\n', on_start(trial));
    
    sendCommand(motor, sprintf("move %d %d", options.end_angle, options.forward_speed));
    waitForMovementDone(motor);
    
    % Record on_end time
    on_end(trial) = toc;
    fprintf('  On end: %.4fs\n', on_end(trial));

    % Stay for half a second
    while toc < on_end(trial) + 0.5
        pause(0.01);
    end
    
    % Lower servo
    if distance_vec(trial) ~= 4
        sendCommand(motor, sprintf("lower %d %d", distance_vec(trial), options.lower_time));
        waitForServoDone(motor);
    end
    
    % Return to start position
    sendCommand(motor, sprintf("move %d %d", options.start_angle, options.return_speed));
    waitForMovementDone(motor);

    fprintf('  Trial %d complete\n\n', trial);

    while toc < off_start(trial) + options.time_per_trial
        pause(0.01);
    end
    
end

% Turn off DAQ trigger
if options.DAQ_flag
    DAQ.write(0);
    fprintf('DAQ trigger ended (0V)\n');
end

% Return to zero position after experiment
fprintf('Experiment complete - returning to 0°...\n');
sendCommand(motor, sprintf("move 0 %d", options.return_speed));
waitForMovementDone(motor);
fprintf('At rest position (0°)\n\n');
fprintf('Total time: %.2f seconds\n', toc);

% Package data
stim_data = options;
stim_data.distance_sequence = distance_vec;
stim_data.off_start = off_start;
stim_data.on_start = on_start;
stim_data.on_end = on_end;
stim_data.monocular_sequence = monocular_sequence;

% Close connection
clear motor;

%% Helper Functions

    function sendCommand(port, cmd)
        % Send command and wait for acknowledgment
        writeline(port, cmd);
        pause(0.05);  % Brief pause for command processing
        
        % Read and display response
        if port.NumBytesAvailable > 0
            response = readline(port);
            % fprintf('    -> %s\n', response);
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