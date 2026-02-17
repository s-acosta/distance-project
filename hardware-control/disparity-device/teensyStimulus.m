function stim_log = teensyStimulus(video_path, opts)
% Play video on Teensy with trial structure and precise timing control
%
% Usage:
%   stim_log = teensyStimulus('stim.mp4')
%   stim_log = teensyStimulus('stim.mp4', 'n_repeats', 5, 'pre_time', 2, 'post_time', 2)
%
% Returns:
%   stim_log.trial_start(i)   - Start time of trial i (seconds)
%   stim_log.pre_start(i)     - Start of PRE phase (gray screen)
%   stim_log.stim_start(i)    - Start of stimulus video
%   stim_log.stim_end(i)      - End of stimulus video
%   stim_log.post_end(i)      - End of POST phase (gray screen)
%   stim_log.frame_times{i}   - Frame timestamps for trial i

arguments
    video_path (1,1) string
    opts.PORT (1,1) string = "COM3"
    opts.baud (1,1) double = 115200
    opts.H (1,1) double = 240
    opts.W (1,1) double = 240
    opts.n_repeats (1,1) double = 2       % Number of trial repetitions
    opts.pre_time (1,1) double = 2.0      % Gray screen before stimulus (seconds)
    opts.post_time (1,1) double = 2.0     % Gray screen after stimulus (seconds)
    opts.target_fps (1,1) double = 10     % Empty = use video's native fps
    opts.resize (1,1) logical = true
    opts.verbose (1,1) logical = true
    opts.DAQ_flag (1,1) logical = false   % Enable DAQ triggering
end

% Validate and open video
assert(isfile(video_path), "Video not found: %s", video_path);
v = VideoReader(video_path);

% Determine target fps
if isempty(opts.target_fps)
    opts.target_fps = v.FrameRate;
end
frame_interval = 1 / opts.target_fps;

% Create gray frame
gray_frame = uint8(ones(opts.H, opts.W, 3) * 127);

if opts.verbose
    fprintf('=== Teensy Trial-Based Stimulus ===\n');
    fprintf('Video: %s | %dx%d @ %.2f fps (%.2f s)\n', ...
        video_path, v.Height, v.Width, v.FrameRate, v.Duration);
    fprintf('Target FPS: %.2f (%.2f ms/frame)\n', opts.target_fps, 1000*frame_interval);
    fprintf('Trial structure:\n');
    fprintf('  PRE:  %.2f s (gray)\n', opts.pre_time);
    fprintf('  STIM: %.2f s (video)\n', v.Duration);
    fprintf('  POST: %.2f s (gray)\n', opts.post_time);
    fprintf('  Total/trial: %.2f s\n', opts.pre_time + v.Duration + opts.post_time);
    fprintf('Number of repeats: %d\n', opts.n_repeats);
    fprintf('Total experiment: %.2f s (%.2f min)\n', ...
        opts.n_repeats * (opts.pre_time + v.Duration + opts.post_time), ...
        opts.n_repeats * (opts.pre_time + v.Duration + opts.post_time) / 60);
    fprintf('DAQ triggering: %s\n', string(opts.DAQ_flag));
    fprintf('====================================\n\n');
end

% Initialize DAQ if requested
if opts.DAQ_flag
    DAQ = daq('ni');
    DAQ.addoutput('Dev1', 'ao0', 'Voltage');
    DAQ.write(0);
    fprintf('DAQ initialized\n');
end

% Open serial
s = serialport(opts.PORT, opts.baud, "Timeout", 2);
configureTerminator(s, "LF");
flush(s);
s.OutputBufferSize = 300000;

% Initialize logging
stim_log = struct();
stim_log.parameters = opts;
stim_log.trial_start = zeros(1, opts.n_repeats);
stim_log.pre_start = zeros(1, opts.n_repeats);
stim_log.stim_start = zeros(1, opts.n_repeats);
stim_log.stim_end = zeros(1, opts.n_repeats);
stim_log.post_end = zeros(1, opts.n_repeats);
stim_log.frame_times = cell(1, opts.n_repeats);

if opts.verbose
    fprintf('Press Enter when ready...\n');
    pause();
end

% Start timing and trigger DAQ
experiment_start = tic;

if opts.DAQ_flag
    DAQ.write(5);
    fprintf('DAQ trigger sent (5V)\n\n');
end

if opts.verbose
    fprintf('Starting experiment...\n\n');
end

% Main trial loop
for trial = 1:opts.n_repeats
    if opts.verbose
        fprintf('--- Trial %d/%d ---\n', trial, opts.n_repeats);
    end
    
    trial_frame_times = [];
    
    % Mark trial start
    stim_log.trial_start(trial) = toc(experiment_start);
    
    % === PRE PHASE (Gray screen) ===
    stim_log.pre_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  PRE start:  %.4f s\n', stim_log.pre_start(trial));
    end
    
    phase_end = stim_log.pre_start(trial) + opts.pre_time;
    while toc(experiment_start) < phase_end
        payload = rgb565_payload(gray_frame);
        writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
        write(s, payload, 'uint8');
        
        trial_frame_times(end+1) = toc(experiment_start);
        
        % Maintain frame rate
        next_frame = stim_log.pre_start(trial) + length(trial_frame_times) * frame_interval;
        while toc(experiment_start) < next_frame
            pause(0.0001);
        end
    end
    
    % === STIMULUS PHASE (Video) ===
    stim_log.stim_start(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  STIM start: %.4f s\n', stim_log.stim_start(trial));
    end
    
    % Reset video to beginning
    v.CurrentTime = 0;
    
    while hasFrame(v)
        rgb = readFrame(v);
        
        if opts.resize && (size(rgb,1) ~= opts.H || size(rgb,2) ~= opts.W)
            rgb = imresize(rgb, [opts.H opts.W], "nearest");
        end
        
        payload = rgb565_payload(rgb);
        writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
        write(s, payload, 'uint8');
        
        trial_frame_times(end+1) = toc(experiment_start);
        
        % Maintain frame rate
        elapsed_in_phase = toc(experiment_start) - stim_log.stim_start(trial);
        target_time = stim_log.stim_start(trial) + (length(trial_frame_times) - sum(trial_frame_times < stim_log.stim_start(trial))) * frame_interval;
        while toc(experiment_start) < target_time
            pause(0.0001);
        end
    end
    
    stim_log.stim_end(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  STIM end:   %.4f s (duration: %.4f s)\n', ...
            stim_log.stim_end(trial), stim_log.stim_end(trial) - stim_log.stim_start(trial));
    end
    
    % === POST PHASE (Gray screen) ===
    post_start = toc(experiment_start);
    phase_end = post_start + opts.post_time;
    
    while toc(experiment_start) < phase_end
        payload = rgb565_payload(gray_frame);
        writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
        write(s, payload, 'uint8');
        
        trial_frame_times(end+1) = toc(experiment_start);
        
        % Maintain frame rate
        next_frame = post_start + sum(trial_frame_times >= post_start) * frame_interval;
        while toc(experiment_start) < next_frame
            pause(0.0001);
        end
    end
    
    stim_log.post_end(trial) = toc(experiment_start);
    if opts.verbose
        fprintf('  POST end:   %.4f s\n', stim_log.post_end(trial));
    end
    
    % Store frame times for this trial
    stim_log.frame_times{trial} = trial_frame_times;
    
    if opts.verbose
        trial_duration = stim_log.post_end(trial) - stim_log.trial_start(trial);
        fprintf('  Trial complete: %d frames, %.2f s (%.2f fps)\n\n', ...
            length(trial_frame_times), trial_duration, length(trial_frame_times)/trial_duration);
    end
end

stim_log.finish_time = toc(experiment_start);

% Turn off DAQ trigger
if opts.DAQ_flag
    DAQ.write(0);
    fprintf('DAQ trigger ended (0V)\n');
end

% Calculate statistics
total_frames = sum(cellfun(@length, stim_log.frame_times));
overall_fps = total_frames / stim_log.finish_time;

if opts.verbose
    fprintf('\n=== Experiment Complete ===\n');
    fprintf('Total time: %.3f s (%.2f min)\n', stim_log.finish_time, stim_log.finish_time/60);
    fprintf('Total frames: %d\n', total_frames);
    fprintf('Overall FPS: %.2f\n', overall_fps);
end

clear s v;
end

%% Helper function
function payload = rgb565_payload(img)
    if ~isa(img,'uint8'), img = im2uint8(img); end
    R5 = bitshift(uint16(img(:,:,1)), -3);
    G6 = bitshift(uint16(img(:,:,2)), -2);
    B5 = bitshift(uint16(img(:,:,3)), -3);
    rgb565 = bitor(bitshift(R5,11), bitor(bitshift(G6,5), B5));
    payload = typecast(reshape(rgb565.', [], 1), 'uint8');
end