function stim_data = linearDeviceControl(arduino_port, options)
% LINEAR_DEVICE_CONTROL Controls servo and stepper motor sequence via Arduino
%   stim_data = linearDeviceControl(arduino_port, options)
%
% Required inputs:
%   arduino_port - COM port where Arduino is connected
%
% Optional parameters (passed as Name-Value pairs):
%   n_repeats - Number of times to repeat the sequence (default: 1)
%   servo_ids - Array of servo IDs to use [0-3] (default: [0,1,2,3])
%   random_flag - Whether to randomize trial order (default: true)
%   trigger_mode - 'DAQ', 'arduino', or 'none' (default: 'none')
%   trigger_port - Port for trigger device if needed
%   save_flag - Whether to save data (default: false)
%   mode - 'history', 'no-history', or 'both' (default: 'both')
%   monocular - Whether to include monocular sequence (default: false)

arguments
    arduino_port char
    options.n_repeats (1,1) double = 1
    options.servo_ids (1,:) double = [0, 3]
    options.random_flag (1,1) logical = true
    options.trigger_mode char {mustBeMember(options.trigger_mode,['DAQ','arduino','none'])} = 'none'
    options.trigger_port = []
    options.save_flag (1,1) logical = false
    options.mode char {mustBeMember(options.mode,['history','no-history','both'])} = 'both'
    options.monocular (1,1) logical = false
end

% Check if monocular option is enabled
if options.monocular
    % Check if n_repeats is a multiple of 4
    if mod(options.n_repeats, 4) ~= 0
        original_repeats = options.n_repeats;
        % Round up to nearest multiple of 4
        options.n_repeats = ceil(options.n_repeats / 4) * 4;
        warning(['For proper monocular sequence blocks, n_repeats must be a multiple of 4. ' ...
                'Adjusting from %d to %d repeats.'], original_repeats, options.n_repeats);
    end
    
    % Recalculate and show total number of trials
    switch options.mode
        case {'history', 'no-history'}
            total_trials = length(options.servo_ids) * options.n_repeats;
        case 'both'
            total_trials = length(options.servo_ids) * 2 * options.n_repeats;
    end
    
    fprintf('Using %d repeats for a total of %d trials\n', options.n_repeats, total_trials);
end

% Setup trigger if requested
if strcmp(options.trigger_mode, 'DAQ')
    DAQ = daq('ni');
    DAQ.addoutput('Dev1', 'ao0', 'Voltage');
    DAQ.write(0);
elseif strcmp(options.trigger_mode, 'arduino')
    trigger_arduino = arduino(options.trigger_port, 'Uno');
    trigger_arduino.writeDigitalPin("D2", 0);
end

% Setup Arduino serial connection
serial_obj = serialport(arduino_port, 115200);
configureTerminator(serial_obj, "LF");
while ~contains(serial_obj.readline(),  "Arduino is ready")
    disp(readline(serial_obj));
end
flush(serial_obj);

% Initialize data storage in UserData
serial_obj.UserData.Time = [];
serial_obj.UserData.Steps = [];
serial_obj.UserData.ServoID = [];
serial_obj.UserData.MiniServoActive = [];

% Set up callback function for data reading
configureCallback(serial_obj, "terminator", @readArduinoData);

% Define trial timing constants
HISTORY_TRIAL_TIME = 15;    % seconds per history trial
NO_HISTORY_TRIAL_TIME = 12;  % seconds per no-history trial

% Set up trial sequences
n_ids = length(options.servo_ids);  % Number of servo IDs

switch options.mode
    case 'history'
        base_servo_sequence = repmat(options.servo_ids, 1);
        base_history_sequence = true(1, length(base_servo_sequence));

    case 'no-history'
        base_servo_sequence = repmat(options.servo_ids, 1);
        base_history_sequence = false(1, length(base_servo_sequence));

    case 'both'
        % Create base sequence with each servo ID in both history and no-history
        base_servo_sequence = repelem(options.servo_ids, 2);
        base_history_sequence = repmat([true false], 1, length(options.servo_ids));
end

servo_sequence = [];
history_sequence = [];

for i = 1:options.n_repeats
    if options.random_flag
        perm = randperm(length(base_servo_sequence));
        servo_sequence = [servo_sequence, base_servo_sequence(perm)];
        history_sequence = [history_sequence, base_history_sequence(perm)];
    else
        servo_sequence = [servo_sequence, base_servo_sequence];
        history_sequence = [history_sequence, base_history_sequence];
    end
end

% Create monocular sequence if enabled
n_trials = length(servo_sequence);
monocular_sequence = [];

if options.monocular
    % Since we ensured n_repeats is a multiple of 4, we can create a pattern
    % that repeats the base_sequence exactly once for each quarter
    
    % Calculate how many repeats constitute one quarter of the total
    quarter_repeats = options.n_repeats / 4;
    
    % Create sequence with each quarter having a specific monocular status
    switch options.mode
        case {'history', 'no-history'}
            % Calculate quarter length in trials
            quarter_length = length(options.servo_ids) * quarter_repeats;
            
            % Create sequence blocks
            monocular_sequence = [
                true(1, quarter_length), ...    % First quarter: true
                false(1, quarter_length), ...   % Second quarter: false
                true(1, quarter_length), ...    % Third quarter: true
                false(1, quarter_length)        % Fourth quarter: false
            ];
        
        case 'both'
            % Calculate quarter length in trials
            quarter_length = length(options.servo_ids) * 2 * quarter_repeats;
            
            % Create sequence blocks
            monocular_sequence = [
                true(1, quarter_length), ...    % First quarter: true
                false(1, quarter_length), ...   % Second quarter: false
                true(1, quarter_length), ...    % Third quarter: true
                false(1, quarter_length)        % Fourth quarter: false
            ];
    end
else
    % If not enabled, create a sequence of all false (no monocular)
    monocular_sequence = false(1, n_trials);
end

% Calculate total time and display estimates
total_time = sum(history_sequence) * HISTORY_TRIAL_TIME + ...
            sum(~history_sequence) * NO_HISTORY_TRIAL_TIME;

% Display time and trial estimates
fprintf('\nTrial Summary:\n');
fprintf('Total trials: %d\n', n_trials);
fprintf('History trials: %d\n', sum(history_sequence));
fprintf('No-history trials: %d\n', sum(~history_sequence));

if options.monocular
    fprintf('Monocular sequence switches:\n');
    
    % Find where the sequence changes (transitions between true/false)
    transitions = find(diff([false, monocular_sequence]) ~= 0);
    
    % Display the transitions in a readable format
    for i = 1:length(transitions)
        trial_num = transitions(i);
        is_monocular = monocular_sequence(trial_num);
        
        if is_monocular
            fprintf('  Trial %3d: Switch to monocular\n', trial_num);
        else
            fprintf('  Trial %3d: Switch to binocular\n', trial_num);
        end
    end
    
    fprintf('Monocular trials: %d, Binocular trials: %d\n', sum(monocular_sequence), sum(~monocular_sequence));
end

fprintf('\nTime Estimates:\n');
fprintf('Total duration: %.2f minutes (%.2f seconds)\n', total_time/60, total_time);
fprintf('History trials: %d (%.2f minutes)\n', sum(history_sequence), sum(history_sequence) * HISTORY_TRIAL_TIME/60);
fprintf('No-history trials: %d (%.2f minutes)\n', sum(~history_sequence), sum(~history_sequence) * NO_HISTORY_TRIAL_TIME/60);

fprintf('\nPress enter when ready...\n');
pause();

% Start trigger if enabled
if strcmp(options.trigger_mode, 'DAQ')
    DAQ.write(5);
elseif strcmp(options.trigger_mode, 'arduino')
    trigger_arduino.writeDigitalPin("D2", 1);
end

% Initialize timing variables 
experiment_start_time = tic;  % Global experiment timer
next_trial_start_time = 0;     % Next scheduled trial start time (in seconds)
trial_start_time = zeros(1, n_trials);
trial_end_time = zeros(1, n_trials);

% Process trials with fixed-interval timing
for i_trial = 1:n_trials
    % Record actual trial start time
    trial_elapsed = toc(experiment_start_time);
    trial_start_time(i_trial) = trial_elapsed;

    % Send command to Arduino
    if history_sequence(i_trial)
        cmd = sprintf('%d,h', servo_sequence(i_trial));
    else
        cmd = sprintf('%d,n', servo_sequence(i_trial));
    end
    writeline(serial_obj, cmd);
    
    % Track trial progress
    if history_sequence(i_trial)
        mode_str = 'history';
        trial_duration = HISTORY_TRIAL_TIME;
    else
        mode_str = 'no history';
        trial_duration = NO_HISTORY_TRIAL_TIME;
    end
    
     % Get monocular status for display
    if options.monocular
        if monocular_sequence(i_trial)
            eye_mode = 'monocular';
        else
            eye_mode = 'binocular';
        end
    else
        eye_mode = '';
    end
    
    % Display trial info with monocular status if enabled
    if options.monocular
        fprintf('Trial %d/%d: Servo %d (%s mode, %s)\n', ...
            i_trial, n_trials, ...
            servo_sequence(i_trial), ...
            mode_str, ...
            eye_mode);
    else
        fprintf('Trial %d/%d: Servo %d (%s mode)\n', ...
            i_trial, n_trials, ...
            servo_sequence(i_trial), ...
            mode_str);
    end
    
    % Calculate when this trial should end to maintain fixed intervals
    if i_trial == 1
        % First trial - start from the current time
        next_trial_start_time = trial_duration;
    else
        % Subsequent trials - use the scheduled next trial start time
        next_trial_start_time = next_trial_start_time + trial_duration;
    end
    
    % Continuously update data until it's time for the next trial
    while toc(experiment_start_time) < next_trial_start_time
        serial_obj = readArduinoData(serial_obj);
    end

    % Record trial end time
    trial_end_time(i_trial) = toc(experiment_start_time);
end

% Stop trigger
if strcmp(options.trigger_mode, 'DAQ')
    DAQ.write(0);
elseif strcmp(options.trigger_mode, 'arduino')
    trigger_arduino.writeDigitalPin("D2", 0);
end

% Cleanup and package data
configureCallback(serial_obj, "off");
stim_data = struct();
stim_data.time = serial_obj.UserData.Time;
stim_data.stepper_pos = serial_obj.UserData.Steps;
stim_data.servo_id = serial_obj.UserData.ServoID;
stim_data.mini_servo_active = serial_obj.UserData.MiniServoActive;
stim_data.trial_sequence = servo_sequence;
stim_data.history_sequence = history_sequence;
stim_data.trial_start_time = trial_start_time;
stim_data.trial_end_time = trial_end_time;

% Include monocular sequence in output data if enabled
if options.monocular
    stim_data.monocular_sequence = monocular_sequence;
end

delete(serial_obj);

if options.save_flag
    current_time = datetime('now');
    save_name = sprintf('stimulus_data_%s.mat', ...
        string(current_time, 'yyyy-MM-dd_HH-mm-ss'));
    save(save_name, 'stim_data');
end
end

function src = readArduinoData(src, ~)
% Callback function to read and parse Arduino data

data = readline(src);

% Parse status message if valid format
if startsWith(data, '<') 
    data = extractBetween(data, '<', '>');
    
    % Extract values using regexp
    stepper_pos = str2double(regexp(data, 'm(\d+)i', 'tokens', 'once'));
    servo_id = str2double(regexp(data, 'i(-?\d+)a', 'tokens', 'once'));
    mini_active = str2double(regexp(data, 'a(\d+)t', 'tokens', 'once'));
    timestamp = str2double(regexp(data, 't(\d+)', 'tokens', 'once'));
    
    % Store data in UserData
    src.UserData.Time(end+1) = timestamp;
    src.UserData.Steps(end+1) = stepper_pos;
    src.UserData.ServoID(end+1) = servo_id;
    src.UserData.MiniServoActive(end+1) = mini_active;
end
end