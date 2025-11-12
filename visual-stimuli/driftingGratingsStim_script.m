% % Equavilent to goard lab's passive drifting gratings
% clear;
% close all;
% sca;
% 
% pre_time = 2;
% post_time = 2;
% on_time = 2;
% screen_id = 2;
% n_repeats = 1;
% DAQ_flag = false;
% 
% options = struct();
% options.screen_id  = 2; % Screen ID for presentatio
% options.n_orientations = 2; % Orientation pool
% options.n_repeats = 1; % Repeats of each presentation
% options.pre_time = pre_time; % seconds preceding stimulus
% options.on_time = on_time; % seconds stimulus presentation
% options.post_time = post_time; % seconds following stimulus
% options.DAQ_flag = false; % For triggering the microscope
% options.random_flag  = false; % Whether to randomize orientation order within each repeat
% 
% orientations = linspace(0, 360-360/options.n_orientations, options.n_orientations);
% 
% orientation_sequence = [];
% for r = 1:options.n_repeats
%     if options.random_flag
%         % Create a randomized order for this repeat
%         perm = randperm(options.n_orientations);
%         orientation_sequence = [orientation_sequence, orientations(perm)];
%     else
%         % Use the original order
%         orientation_sequence =  [orientation_sequence, orientations];
%     end
% end
% 
% % Store the sequence in the output structure
% stim_data.orientation_sequence = orientation_sequence;
% 
% % Create stimulus objects
% 
% for i = 1:length(orientation_sequence)
%     visual_stimulus(i) = DriftingGrating(orientation_sequence(i));
% end
% 
% % Calculate the time it will take to present the stimulus
% total_time = options.pre_time + options.on_time + options.post_time;
% fprintf('Total number of trials: %d\n', options.n_repeats * options.n_orientations);
% fprintf('Stimulus presentation will take %.2f seconds\n', total_time);
% fprintf('Total time for all repeats: %.2f seconds\n', total_time * options.n_repeats);
% 
% pause()
% 
% % Instantiate and configure stimulus manager
% 
% manager = StimulusManager(visual_stimulus);
% manager.setScreenID(options.screen_id);
% manager.initialize();
% manager.setTrigger(options.DAQ_flag);
% %         start_time = tic;
% 
% % Start presentation
% manager.start();
% 
% for p = 1:length(visual_stimulus)
%     manager.present(p)
% end
% 
% %     % Present the stimulus according to the sequence
% %     for  i = 1:length(orientation_sequence)
% %
% %         % Record the start time of the pre-time
% %         stim_data.pre_time_start(i) = toc(start_time);
% %         stim_data.on_time_start(i) = toc(start_time)+2;
% %
% %         % Present the stimulus
% %         manager.present(i);
% %
% %         % Record the end time of the post-time
% % %         stim_data.post_time_start(i) = toc(start_time);
% %
% %     end
% 
% % Clean up
% %         finish_time = toc(start_time);
% manager.finish();
% 


clear;
close all;
sca;

n_repeats = 1; % 10 repeats of each group of presentations
n_orientations = 12;
random_flag = false;
pre_time = 2; % seconds preceding stimulus on
on_time = 2; % seconds stimulus presentation
post_time = 2; % seconds following stimulus on

DAQ_flag = 0; % For triggering the microscope

% orientations = linspace(0, 360-360/n_orientations, n_orientations);
% 
% orientation_sequence = [];
% for r = 1:n_repeats
%     if random_flag
%         % Create a randomized order for this repeat
%         perm = randperm(n_orientations);
%         orientation_sequence = [orientation_sequence, orientations(perm)];
%     else
%         % Use the original order
%         orientation_sequence =  [orientation_sequence, orientations];
%     end
% end

orientation_list = 0:30:330;
n_presentations = length(orientation_list);

for ori = 1:numel(orientation_list)
    stimulus(ori) = DriftingGrating(orientation_list(ori));
end

% Instantiate objects
window = Screen('OpenWindow', 1, WhiteIndex(1)/2);
manager = StimulusManager(stimulus);
manager.setScreenID(1);


manager.initialize();
manager.setTrigger(DAQ_flag); 

manager.start();

% Present the stimulus
for p = 1:numel(orientation_list)
    manager.present(p);
    imageArray = Screen('GetImage', windowPtr);
end

manager.finish();
