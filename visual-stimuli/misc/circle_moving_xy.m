% ==== 240x240 window; horizontal motion; save frames ====
sca; close all; clear;
PsychDefaultSetup(2);
Screen('Preference','VisualDebugLevel', 1);
Screen('Preference','SkipSyncTests', 1);

% Screen + colors
screens      = Screen('Screens');
screenNumber = max(screens);
white        = WhiteIndex(screenNumber);
black        = BlackIndex(screenNumber);
grey         = white/2;

% Center a 240x240 window on the chosen screen
screenRect   = Screen('Rect', screenNumber);
[scx, scy]   = RectCenter(screenRect);
winRect      = CenterRectOnPointd([0 0 240 240], scx, scy);
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey, winRect);
[~, yCenter] = RectCenter(windowRect);

% Enable alpha blending
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% ----- Build the checker-circle texture (same look as before) -----
texSize   = 240;
rcycles   = 1;          % radial cycles
tcycles   = 2;          % angular cycles
xylim     = pi * rcycles;

[x, y]    = meshgrid(linspace(-xylim, xylim, texSize));
at        = atan2(y, x);
checksN   = (1 + sign(sin(at * tcycles) + eps) .* sign(sin(sqrt(x.^2 + y.^2))))/2;
circle    = (x.^2 + y.^2) <= xylim^2;
checksN   = circle .* checksN + 0.5 * ~circle;        % grey outside circle

texImg    = uint8(round(checksN * 255));
texRGB    = cat(3, texImg, texImg, texImg);
radialCheckerboardTexture = Screen('MakeTexture', window, texRGB);

% Stimulus size (diameter of the moving circle on screen, in px)
stimDiam  = 13;                               % tweak as needed
baseRect  = [0 0 stimDiam stimDiam];

% Timing + saving setup
ifi        = Screen('GetFlipInterval', window);
timeToEdge = 2;                                % seconds for center: x=0 -> x=240
cycleDur   = 2 * timeToEdge;                   % 0->240->0 ping-pong
tStart     = GetSecs;
vbl        = Screen('Flip', window);

doSave     = true;
saveSecs   = 5;                                % run & save ~5 s (or until keypress)
outDir     = fullfile(pwd, 'stim_frames');
if doSave && ~exist(outDir, 'dir'), mkdir(outDir); end
frameIdx   = 0;
lastImg    = [];

try
    while (GetSecs - tStart) < saveSecs && ~KbCheck
        % ----- Compute horizontal center position p in [0..240] -----
        t = mod(GetSecs - tStart, cycleDur);
        if t <= timeToEdge
            p = (t / timeToEdge) * 240;               % 0 -> 240
        else
            p = ((cycleDur - t) / timeToEdge) * 240;  % 240 -> 0
        end

        % Center at (p, yCenter). (At p=0 or 240, part may clip at edges.)
        dstRect = CenterRectOnPointd(baseRect, p, yCenter);

        % ----- Draw frame -----
        Screen('FillRect', window, grey);            % clear bg
        Screen('DrawTexture', window, radialCheckerboardTexture, [], dstRect);
        vbl = Screen('Flip', window, vbl + 0.5 * ifi);

        % Save the displayed frame
        if doSave
            frameIdx = frameIdx + 1;
            img = Screen('GetImage', window);        % reads front buffer
            lastImg = img;
            imwrite(img, fullfile(outDir, sprintf('frame_%04d.png', frameIdx)));
        end

        WaitSecs('YieldSecs', 0.002);
    end
catch ME
    sca;
    rethrow(ME);
end

% Save a convenient still of the last frame
if doSave && ~isempty(lastImg)
    imwrite(lastImg, fullfile(pwd, 'stim_lastframe.png'));
end

% Cleanup
Screen('Close', radialCheckerboardTexture);
sca;

%%
% ==== 240x240 window; horizontal motion; save to video ====
sca; close all; clear;
PsychDefaultSetup(2);
Screen('Preference','VisualDebugLevel', 1);
Screen('Preference','SkipSyncTests', 1);

% Screen + colors
screens      = Screen('Screens');
screenNumber = max(screens);
white        = WhiteIndex(screenNumber);
black        = BlackIndex(screenNumber);
grey         = white/2;

% Center a 240x240 window on the chosen screen
screenRect   = Screen('Rect', screenNumber);
[scx, scy]   = RectCenter(screenRect);
winRect      = CenterRectOnPointd([0 0 240 240], scx, scy);
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey, winRect);
[~, yCenter] = RectCenter(windowRect);

% Enable alpha blending
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% ----- Build the checker-circle texture (same look as before) -----
texSize   = 240;
rcycles   = 1;          % radial cycles
tcycles   = 2;          % angular cycles
xylim     = pi * rcycles;

[x, y]    = meshgrid(linspace(-xylim, xylim, texSize));
at        = atan2(y, x);
checksN   = (1 + sign(sin(at * tcycles) + eps) .* sign(sin(sqrt(x.^2 + y.^2))))/2;
circle    = (x.^2 + y.^2) <= xylim^2;
checksN   = circle .* checksN + 0.5 * ~circle;        % grey outside circle

texImg    = uint8(round(checksN * 255));
texRGB    = cat(3, texImg, texImg, texImg);
radialCheckerboardTexture = Screen('MakeTexture', window, texRGB);

% Stimulus size (diameter of the moving circle on screen, in px)
stimDiam  = 13;                               % tweak as needed
baseRect  = [0 0 stimDiam stimDiam];

% Timing + saving setup
ifi        = Screen('GetFlipInterval', window);
timeToEdge = 2;                                % seconds for center: x=0 -> x=240
cycleDur   = 2 * timeToEdge;                   % 0->240->0 ping-pong
tStart     = GetSecs;
vbl        = Screen('Flip', window);

% ---- Video writer instead of per-frame PNGs ----
saveSecs   = 5;                                % run & save ~5 s (or until keypress)
fps        = max(1, round(1/ifi));             % approximate display fps
vidPath    = fullfile(pwd, 'stim.mp4');
vid        = VideoWriter(vidPath, 'MPEG-4');   % or 'Motion JPEG AVI'
vid.FrameRate = fps;
vid.Quality   = 95;                            % 0–100 (MPEG-4 uses this)
open(vid);

lastImg = [];                                  % keep last frame as still (optional)

try
    while (GetSecs - tStart) < saveSecs && ~KbCheck
        % ----- Compute horizontal center position p in [0..240] -----
        t = mod(GetSecs - tStart, cycleDur);
        if t <= timeToEdge
            p = (t / timeToEdge) * 240;               % 0 -> 240
        else
            p = ((cycleDur - t) / timeToEdge) * 240;  % 240 -> 0
        end

        % Center at (p, yCenter). (At p=0 or 240, part may clip at edges.)
        dstRect = CenterRectOnPointd(baseRect, p, yCenter);

        % ----- Draw frame -----
        Screen('FillRect', window, grey);            % clear bg
        Screen('DrawTexture', window, radialCheckerboardTexture, [], dstRect);
        vbl = Screen('Flip', window, vbl + 0.5 * ifi);

        % Grab the displayed frame and write to video
        img     = Screen('GetImage', window);        % reads front buffer
        lastImg = img;
        writeVideo(vid, img);

        WaitSecs('YieldSecs', 0.002);
    end
catch ME
    % Ensure resources close even on error
    try close(vid); catch, end
    sca;
    rethrow(ME);
end

% Optional: also save a convenient still of the last frame
if ~isempty(lastImg)
    imwrite(lastImg, fullfile(pwd, 'stim_lastframe.png'));
end

% Cleanup
close(vid);
Screen('Close', radialCheckerboardTexture);
sca;

fprintf('Saved video to: %s\n', vidPath);
