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

arguments
    arduino_port char
    options.n_repeats (1,1) double = 1
    options.servo_ids (1,:) double = [0, 3]
    options.random_flag (1,1) logical = true
    options.trigger_mode char {mustBeMember(options.trigger_mode,['DAQ','arduino','none'])} = 'none'
    options.trigger_port = []
    options.save_flag (1,1) logical = false
    options.mode char {mustBeMember(options.mode,['history','no-history','both'])} = 'both'
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
HISTORY_TRIAL_TIME = 17;    % seconds per history trial
NO_HISTORY_TRIAL_TIME = 10;  % seconds per no-history trial

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
        history_sequence = [history_sequence, base_servo_history_sequence];

    end

end

% Calculate total time and display estimates
total_time = sum(history_sequence) * HISTORY_TRIAL_TIME + ...
            sum(~history_sequence) * NO_HISTORY_TRIAL_TIME;

% Display time and trial estimates
fprintf('\nTrial Summary:\n');
fprintf('Total trials: %d\n', length(servo_sequence));
fprintf('History trials: %d\n', sum(history_sequence));
fprintf('No-history trials: %d\n\n', sum(~history_sequence));

fprintf('Time Estimates:\n');
fprintf('Total duration: %.2f minutes (%.2f seconds)\n', total_time/60, total_time);
fprintf('History trials: %d (%.2f minutes)\n', sum(history_sequence), sum(history_sequence) * HISTORY_TRIAL_TIME/60);
fprintf('No-history trials: %d (%.2f minutes)\n\n', sum(~history_sequence), sum(~history_sequence) * NO_HISTORY_TRIAL_TIME/60);

fprintf('Press enter when ready...\n');
pause();

% Start trigger if enabled
if strcmp(options.trigger_mode, 'DAQ')
    DAQ.write(5);
elseif strcmp(options.trigger_mode, 'arduino')
    trigger_arduino.writeDigitalPin("D2", 1);
end

tic;
trial_start_time = zeros(1, length(servo_sequence)); % Initialize trial start time array
trial_end_time = zeros(1, length(servo_sequence)); % Initialize trial end time array

i_trial = 1;
while i_trial <= length(servo_sequence)
    % read input
    serial_obj = readArduinoData(serial_obj);
    trial_start_time(i_trial) = toc;

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
    else
        mode_str = 'no history';
    end
    fprintf('Trial %d/%d: Servo %d (%s mode)\n', ...
        i_trial, length(servo_sequence), ...
        servo_sequence(i_trial), ...
        mode_str);
    
    % Wait for trial completion
    if history_sequence(i_trial)
        wait_time = HISTORY_TRIAL_TIME;
    else
        wait_time = NO_HISTORY_TRIAL_TIME;
    end

    while (toc - trial_start_time(i_trial)) < wait_time
        serial_obj = readArduinoData(serial_obj);
    end

    % Store trial end time
    trial_end_time(i_trial) = toc;
    i_trial = i_trial + 1;
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