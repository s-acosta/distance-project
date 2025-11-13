function stim_log = teensy_play_stimulus_timed(video_path, opts)
% Play video on Teensy with precise frame-by-frame timing control
%
% Usage:
%   stim_log = teensy_play_stimulus_timed('stim.mp4')
%   stim_log = teensy_play_stimulus_timed('stim.mp4', 'PORT', 'COM7', 'target_fps', 30)
%
% Returns:
%   stim_log.frame_times      - Timestamp when each frame was displayed (seconds)
%   stim_log.frame_numbers    - Frame index from video file
%   stim_log.actual_fps       - Achieved frame rate
%   stim_log.total_frames     - Total frames displayed

arguments
    video_path (1,1) string
    opts.PORT (1,1) string = "COM7"
    opts.baud (1,1) double = 115200
    opts.H (1,1) double = 240
    opts.W (1,1) double = 240
    opts.target_fps (1,1) double = 10 % Empty = use video's native fps
    opts.resize (1,1) logical = true
    opts.verbose (1,1) logical = true
end

% Validate and open video
assert(isfile(video_path), "Video not found: %s", video_path);
v = VideoReader(video_path);

% Determine target fps
if isempty(opts.target_fps)
    opts.target_fps = v.FrameRate;
end
frame_interval = 1 / opts.target_fps;

if opts.verbose
    fprintf('Video: %s | %dx%d @ %.2f fps\n', video_path, v.Height, v.Width, v.FrameRate);
    fprintf('Target: %.2f fps (%.2f ms/frame)\n', opts.target_fps, 1000*frame_interval);
end

% Open serial
s = serialport(opts.PORT, opts.baud, "Timeout", 2);
configureTerminator(s, "LF");
flush(s);
s.OutputBufferSize = 300000;

% Pre-allocate logging
max_frames = ceil(v.Duration * opts.target_fps) + 100;
frame_times = zeros(1, max_frames);
frame_numbers = zeros(1, max_frames);

% Main loop
start_time = tic;
frame_idx = 0;
video_frame_idx = 0;
next_frame_time = 0;

if opts.verbose, fprintf('Playing...\n'); end

while hasFrame(v)
    current_time = toc(start_time);
    
    % Check if it's time for next frame
    if current_time >= next_frame_time
        % Read and prepare frame
        rgb = readFrame(v);
        video_frame_idx = video_frame_idx + 1;
        
        if opts.resize && (size(rgb,1) ~= opts.H || size(rgb,2) ~= opts.W)
            rgb = imresize(rgb, [opts.H opts.W], "nearest");
        end
        
        % Send to Teensy
        payload = rgb565_payload(rgb);
        writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
        write(s, payload, 'uint8');
        
        % Log timing
        frame_idx = frame_idx + 1;
        frame_times(frame_idx) = toc(start_time);
        frame_numbers(frame_idx) = video_frame_idx;
        
        % Progress
        if opts.verbose && mod(frame_idx, 60) == 0
            fprintf('Frame %d | %.2f fps\n', frame_idx, frame_idx / toc(start_time));
        end
        
        % Schedule next frame
        next_frame_time = next_frame_time + frame_interval;
    else
        pause(0.001);  % Small wait to avoid busy loop
    end
end

% Trim and package results
duration = toc(start_time);
frame_times = frame_times(1:frame_idx);
frame_numbers = frame_numbers(1:frame_idx);

stim_log = struct();
stim_log.video_path = video_path;
stim_log.parameters = opts;
stim_log.frame_times = frame_times;
stim_log.frame_numbers = frame_numbers;
stim_log.total_frames = frame_idx;
stim_log.duration = duration;
stim_log.actual_fps = frame_idx / duration;
stim_log.target_fps = opts.target_fps;

if opts.verbose
    fprintf('\nComplete: %d frames in %.3f s (%.2f fps)\n', ...
        frame_idx, duration, stim_log.actual_fps);
end

clear s;
end

%% Helper
function payload = rgb565_payload(img)
    if ~isa(img,'uint8'), img = im2uint8(img); end
    R5 = bitshift(uint16(img(:,:,1)), -3);
    G6 = bitshift(uint16(img(:,:,2)), -2);
    B5 = bitshift(uint16(img(:,:,3)), -3);
    rgb565 = bitor(bitshift(R5,11), bitor(bitshift(G6,5), B5));
    payload = typecast(reshape(rgb565.', [], 1), 'uint8');
end