function video_path = generate_leftright_stim(start_point, end_point, stim_size, duration, frame_rate, opts)
% Generate video of radial checkerboard circle moving horizontally
%
% Usage:
%   video_path = generate_leftright_stim(0, 240, 13, 6, 10)
%   video_path = generate_leftright_stim(50, 200, 20, 4, 15, 'output_name', 'my_stim.mp4')
%
% Args:
%   start_point  - Starting x position (pixels)
%   end_point    - Ending x position (pixels)
%   stim_size    - Diameter of circle (pixels)
%   duration     - Video duration (seconds)
%   frame_rate   - Frame rate (Hz)
%
% Optional:
%   output_name  - Output filename (default: auto-generated)
%   window_size  - Display window size (default: 240)
%   rcycles      - Radial cycles in checkerboard (default: 1)
%   tcycles      - Angular cycles in checkerboard (default: 2)
%   elevation    - Fixed y position (pixels) to hold the circle at while it
%                  sweeps in x (default: [], meaning the window's vertical
%                  center -- i.e. the original horizontal-only sweep)

arguments
    start_point (1,1) double
    end_point (1,1) double
    stim_size (1,1) double
    duration (1,1) double
    frame_rate (1,1) double
    opts.output_name (1,1) string = ""
    opts.window_size (1,1) double = 240
    opts.rcycles (1,1) double = 1
    opts.tcycles (1,1) double = 2
    opts.elevation (1,:) double = []
end

% Generate default output name if not provided
if opts.output_name == ""
    if isempty(opts.elevation)
        opts.output_name = sprintf('stim_x%d_to_%d_size%d_%.1fs_%dfps.mp4', ...
            round(start_point), round(end_point), round(stim_size), duration, round(frame_rate));
    else
        opts.output_name = sprintf('stim_x%d_to_%d_size%d_y%d_%.1fs_%dfps.mp4', ...
            round(start_point), round(end_point), round(stim_size), round(opts.elevation), ...
            duration, round(frame_rate));
    end
end

sca; close all;
PsychDefaultSetup(2);
Screen('Preference','VisualDebugLevel', 1);
Screen('Preference','SkipSyncTests', 1);

% Screen + colors
screens      = Screen('Screens');
screenNumber = max(screens);
white        = WhiteIndex(screenNumber);
black        = BlackIndex(screenNumber);
grey         = white/2;

% Center window on screen
screenRect   = Screen('Rect', screenNumber);
[scx, scy]   = RectCenter(screenRect);
winRect      = CenterRectOnPointd([0 0 opts.window_size opts.window_size], scx, scy);
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey, winRect);
[~, yCenter] = RectCenter(windowRect);
if isempty(opts.elevation)
    yPos = yCenter;             % default: sweep along the vertical center
else
    yPos = opts.elevation;      % fixed elevation override
end

% Enable alpha blending
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Build the radial checkerboard texture
texSize   = opts.window_size;
xylim     = pi * opts.rcycles;

[x, y]    = meshgrid(linspace(-xylim, xylim, texSize));
at        = atan2(y, x);
checksN   = (1 + sign(sin(at * opts.tcycles) + eps) .* sign(sin(sqrt(x.^2 + y.^2))))/2;
circle    = (x.^2 + y.^2) <= xylim^2;
checksN   = circle .* checksN + 0.5 * ~circle;        % grey outside circle

texImg    = uint8(round(checksN * 255));
texRGB    = cat(3, texImg, texImg, texImg);
radialCheckerboardTexture = Screen('MakeTexture', window, texRGB);

% Stimulus parameters
baseRect  = [0 0 stim_size stim_size];

% Video parameters
total_frames = duration * frame_rate;

% Output video
[out_dir, ~, ~] = fileparts(opts.output_name);
if out_dir == ""
    video_path = fullfile(pwd, opts.output_name);   % bare filename -> current folder
else
    video_path = char(opts.output_name);            % caller passed a full path -> use as-is
end
vid = VideoWriter(video_path, 'MPEG-4');
vid.FrameRate = frame_rate;
vid.Quality = 95;
open(vid);

fprintf('Generating video:\n');
fprintf('  Motion: x=%d to x=%d\n', round(start_point), round(end_point));
if ~isempty(opts.elevation)
    fprintf('  Elevation (fixed y): %d pixels\n', round(opts.elevation));
end
fprintf('  Stimulus size: %d pixels\n', round(stim_size));
fprintf('  Duration: %.1f s at %d fps (%d frames)\n', duration, round(frame_rate), round(total_frames));

% Timing setup
ifi = Screen('GetFlipInterval', window);
vbl = Screen('Flip', window);

try
    for frame_num = 1:total_frames
        % Calculate horizontal position: start_point to end_point linearly
        t = (frame_num - 1) / max(1, total_frames - 1);  % 0 to 1
        p = start_point + t * (end_point - start_point);
        
        % Center circle at (p, yPos) -- yPos is fixed for the whole video
        dstRect = CenterRectOnPointd(baseRect, p, yPos);
        
        % Draw frame
        Screen('FillRect', window, grey);
        Screen('DrawTexture', window, radialCheckerboardTexture, [], dstRect);
        vbl = Screen('Flip', window, vbl + 0.5 * ifi);
        
        % Capture and write to video
        img = Screen('GetImage', window);
        writeVideo(vid, img);
        
        % Progress indicator
        if mod(frame_num, max(1, round(total_frames/10))) == 0
            fprintf('  Progress: %d%%\n', round(100*frame_num/total_frames));
        end
    end
catch ME
    try close(vid); catch, end
    sca;
    rethrow(ME);
end

% Cleanup
close(vid);
Screen('Close', radialCheckerboardTexture);
sca;

fprintf('\nVideo saved to: %s\n', video_path);
fprintf('Parameters: x%d→%d, size=%d, %.1fs @ %dfps\n', ...
    round(start_point), round(end_point), round(stim_size), duration, round(frame_rate));

end