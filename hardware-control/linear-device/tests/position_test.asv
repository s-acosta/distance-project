%% Position test
% This scripts is to test that the device moves to the desired position
% when prompted with Matlab

stepper_port = 'COM10';
stepper = serialport(stepper_port, 115200);
configureTerminator(stepper, "LF");
while ~contains(stepper.readline(),  "Arduino is ready")
end

stepper.flush();
stepper.UserData = struct("Steps",[],"Time", []);
configureCallback(stepper,"terminator", @readStepsCount)

tic;

%% position to go
tic;
goal_position = 0;

order = strcat("<", num2str(goal_position), ">");
stepper.writeline(order);

stop = 0;
while ~stop
    [stepper, stop] = readStepsCount(stepper);
end

% Return
goal_position = 1110;
order = strcat("<", num2str(goal_position), ">");
stepper.writeline(order);
stop = 0;
while ~stop
    [stepper, stop] = readStepsCount(stepper);
end
disp(toc)

%%
configureCallback(stepper, "off")
data = struct;
data.time = stepper.UserData.Time;
data.steps = stepper.UserData.Steps;

plot(data.time, data.steps)

clear stepper