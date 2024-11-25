% %% Position test
% a = arduino('com6', 'Uno');
% a.writeDigitalPin("D2", 0);

DAQ = daq('ni');
DAQ.addoutput('Dev1', 'ao1', 'Voltage');
DAQ.write(0);

%%
orders = [2, 1, 2, 1];
stepper_port = 'com11';

stepper = serialport(stepper_port, 115200);
configureTerminator(stepper, "LF");
while ~contains(stepper.readline(),  "Arduino is ready")
end

%% position to go
stepper.flush();
stepper.UserData = struct("Steps",[],"Time", []);


n_trials = length(orders);

is_triggered = false;

i_trial = 0;
while i_trial < n_trials

    if is_triggered == false
        DAQ.write(5);
    end

    i_trial = i_trial + 1;
    order = strcat("<", num2str(orders(i_trial)), ">");
    stepper.writeline(order);

    [stepper, stop] = readStepsCount(stepper);
    while stop == 0
         [stepper, stop] = readStepsCount(stepper);
    end

end

% a.writeDigitalPin("D2", 0);
DAQ.write(0);
configureCallback(stepper, "off")
data = struct;
data.orders = orders;
data.time = stepper.UserData.Time;
data.steps = stepper.UserData.Steps;
    
%%

order = strcat("<", num2str(100), ">");

% is_moving = 0;
% is_coming_back = 0;
% stop = 0;
% i_trial = 0;
% 
% trial_position = [end_position, start_position];
% 
% while i_trial < 2
% 
%     [stepper, stop] = readStepsCount(stepper);
%     
%     if is_moving == 0
%         
%         order = strcat("<", num2str(trial_position(i_trial+1)), ">");
%         stepper.writeline(order);
%         is_going = 1;
% 
%     end
% 
%     if stop == 1
% 
%         i_trial = i_trial + 1;
%         is_going = 0;
% 
%     end
% 
% end
% 
configureCallback(stepper, "off")
data = struct;
data.time = stepper.UserData.Time;
data.steps = stepper.UserData.Steps;


%%
plot(data.time./1000, data.steps)

clear steppe

