%% Position test
n_trials = 1;
start_position = 0;
end_position = 700;
off_time = 0;

stepper = setup_stepper('COM3');

%% position to go
tic;

stepper.flush();
stepper.UserData = struct("Steps",[],"Time", []);
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

