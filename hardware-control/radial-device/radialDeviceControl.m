function stim_data = radialDeviceControl(encoder_port, motor_port, options)

% Variable set-up
arguments
    encoder_port
    motor_port
    options.n_repeats = 2;
    options.distances = [1, 2, 3, 4];
    options.off_time = 1;
    options.DAQ_flag = false;
    options.random_flag = true;
    options.save_flag = false;
    options.monocular = false;
end

if nargin < 1 || isempty(motor_port)
    temp = inputdlg('Input COM port of arduino: ');
    motor_port = temp{:};
end

%% Initialize DAQ
if options.DAQ_flag
    DAQ = daq('ni');
    DAQ.addoutput('Dev1', 'ao1', 'Voltage');
    DAQ.write(0);
end

%% Configuring Motor
motor = serialport(motor_port, 115200);
configureTerminator(motor, "LF");
while ~contains(motor.readline(),  "Arduino is ready")
end

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


% distance_vec = repmat(options.distances, [1 options.n_repeats]);
%
% if options.random_flag
%     distance_vec = distance_vec(randperm(length(distance_vec)));
% end

off_start = zeros(size(distance_vec));
on_start = zeros(size(distance_vec));
on_end = zeros(size(distance_vec));


%% Configuring Encoder
encoder = serialport(encoder_port, 115200);
configureTerminator(encoder, "LF");
while ~contains(encoder.readline(),  "Arduino is ready")
end

%% Making it go to start point
order = strcat("<", num2str(5), ">");
motor.writeline(order);
motor.flush()
while ~contains(motor.readline(), "Movement Done")
end
while ~contains(motor.readline(), "Movement Done")
end

%% Waiting for user input to go ahead
time_of_stimulus = options.n_repeats * n_distances * 11;
frames_of_stimulus = time_of_stimulus * 10;

fprintf('Frames of stimulus: %i \n', frames_of_stimulus);
fprintf('Time of stimulus: %.2f minutes \n', time_of_stimulus/60);
fprintf('Make sure it looks good before starting... \n')

% If monocular mode, inform user when to switch between monocular and binocular
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
    fprintf('Trials per condition and distance %d\n', sum(monocular_sequence)/length(options.distances))
end

fprintf('Press enter when ready \n')

pause()

%% Trigger encoder and 2p
encoder.flush();
encoder.UserData = struct("Angle",[],"Time", [], "Count",1);
configureCallback(encoder,"terminator", @readStreamValues);
encoder.UserData.Time = [];

tic;
if options.DAQ_flag
    DAQ.write(5);
end

for i = 1:length(distance_vec)

    fprintf('\tPresentation %d | Distance %d\n', i, distance_vec(i));
    % pause 2 secs before trial

    off_start(i) = toc;
    fprintf('\t\tOff start: %0.04fs\n', off_start(i));
    while toc < off_start(i) + options.off_time
        encoder = readStreamValues(encoder);
    end

    on_start(i) = toc;
    fprintf('\t\tOn start: %0.04fs\n', on_start(i));
    order = strcat("<", num2str(distance_vec(i)), ">");
    motor.writeline(order);
    while ~contains(motor.readline(), "Movement Done")
        encoder = readStreamValues(encoder);
    end
    on_end(i) = toc;
    fprintf('\t\tOn end: %0.04fs\n', on_end(i));

    while ~contains(motor.readline(), "Movement Done")
        encoder = readStreamValues(encoder);
    end

    while toc < floor(off_start(i) + 11)
    end

end
configureCallback(encoder, "off")

if options.DAQ_flag
    DAQ.write(0);
end

stim_data = options;
stim_data.distance = distance_vec;

stim_data.off_start = off_start;
stim_data.on_start = on_start;
stim_data.on_end = on_end;
stim_data.encoder_time = encoder.UserData.Time;
stim_data.encoder_time = stim_data.encoder_time - stim_data.encoder_time(1);
stim_data.encoder_angle = encoder.UserData.Angle;
stim_data.monocular_sequence = monocular_sequence;

% if options.save_flag
%     uisave('stim_data', sprintf('%s_stimulus_data_distance.mat', date));
% end

end

function src = readStreamValues(src, ~)

% Read the ASCII data from the serialport object.
data = readline(src);

if data == "S"
    configureCallback(src, "off");
end

if ~any(regexp(data ,'[0-9]'))
    fprintf(strcat(data, ' \n'))
    return
end

marker = data{1}(1);
data = str2double(data{1}(2:end));

switch marker
    case 't'
        time = data;
        %          fprintf('Time is %i miliseconds \n', time);
        src.UserData.Time(end+1) = time;
    case 'c'
        angle = 0.375 * data;
        % fprintf('Steps are %.2f \n', data)
        src.UserData.Angle(end+1) = angle;
end

% Convert the string data to numeric type and save it in the UserData
% property of the serialport object.


end