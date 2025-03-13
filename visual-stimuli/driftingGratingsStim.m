function stim_data = driftingGratingsStim(options)
% DRIFTING_GRATING_STIMULUS Presents drifting grating visual stimuli
%
% USAGE:
%   stim_data = drifting_grating_stimulus()
%   stim_data = drifting_grating_stimulus('orientation_list', [0,45,90,135])
%
% INPUTS:
%   options - Name-value pairs for stimulus parameters
%     .screen_id - Screen ID for presentation (default: 4)
%     .orientation_list - List of orientations in degrees (default: [0,90])
%     .n_repeats - Number of repetitions for each orientation (default: 1)
%     .pre_time - Time before stimulus in seconds (default: 2)
%     .on_time - Stimulus presentation time in seconds (default: 2)
%     .post_time - Time after stimulus in seconds (default: 2)
%     .DAQ_flag - Flag for triggering data acquisition (default: 0)
%     .random_flag - Whether to randomize orientation order within each repeat (default: true)
%
% OUTPUTS:
%   stim_data - Structure containing stimulus parameters and timing information
%
% See also: DriftingGrating, StimulusManager

    arguments
        options.screen_id (1,1) double = 4 % Screen ID for presentation
        options.n_orientations (1,1) double = 2 % Orientation pool
        options.n_repeats (1,1) double {mustBePositive, mustBeInteger} = 2 % Repeats of each presentation
        options.pre_time (1,1) double {mustBeNonnegative} = 2 % seconds preceding stimulus
        options.on_time (1,1) double {mustBePositive} = 2 % seconds stimulus presentation
        options.post_time (1,1) double {mustBeNonnegative} = 2 % seconds following stimulus
        options.DAQ_flag (1,1) logical = false % For triggering the microscope
        options.random_flag (1,1) logical = false % Whether to randomize orientation order within each repeat
    end

    % Extract parameters from options
    orientations = linspace(0, 360-360/options.n_orientations, options.n_orientations);
    
    % Initialize output structure
    stim_data = struct();
    stim_data.parameters = options;
    
    % Create the presentation sequence with randomization if requested
    stim_data.pre_time_start = zeros(1, options.n_repeats * options.n_orientations);
    stim_data.on_time_start = zeros(1, options.n_repeats * options.n_orientations);
    stim_data.post_time_start = zeros(1, options.n_repeats * options.n_orientations);
    
    orientation_sequence = [];
    for r = 1:options.n_repeats
        if options.random_flag
            % Create a randomized order for this repeat
            perm = randperm(options.n_orientations);
            orientation_sequence = [orientation_sequence, orientations(perm)];
        else
            % Use the original order
            orientation_sequence =  [orientation_sequence, orientations];
        end
    end

    % Store the sequence in the output structure
    stim_data.orientation_sequence = orientation_sequence;
    
    % Create stimulus objects
    try
        for i = 1:length(orientation_sequence)
                visual_stimulus(i) = DriftingGrating(orientation_sequence(i));
        end
    catch ME
        error('Failed to create DriftingGrating objects: %s', ME.message);
    end

    % Calculate the time it will take to present the stimulus
    total_time = options.pre_time + options.on_time + options.post_time;
    fprintf('Total number of trials: %d\n', options.n_repeats * options.n_orientations);
    fprintf('Stimulus presentation will take %.2f seconds\n', total_time);
    fprintf('Total time for all repeats: %.2f seconds\n', total_time * options.n_repeats);
    
    pause()

    % Instantiate and configure stimulus manager
    try
        manager = StimulusManager(visual_stimulus);
        manager.setScreenID(options.screen_id);
        manager.initialize();
        manager.setTrigger(options.DAQ_flag);
        start_time = tic;
        
        % Start presentation
        manager.start();
        
        % Present the stimulus according to the sequence
        for  i = 1:length(orientation_sequence)
           
                % Record the start time of the pre-time
                stim_data.pre_time_start(i) = toc(start_time);
                stim_data.on_time_start(i) = toc(start_time)+2;

                % Present the stimulus
                manager.present(i);

                % Record the end time of the post-time
                stim_data.post_time_start(i) = toc(start_time);
                
        end
        
        % Clean up
         finish_time = toc(start_time);
        manager.finish();

end
