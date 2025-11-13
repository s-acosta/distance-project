% Clear the workspace and the screen
sca; close all; clear;

% Default Psychtoolbox setup
PsychDefaultSetup(2);

% Get screen and colors
screens       = Screen('Screens');
screenNumber  = max(screens);
white         = WhiteIndex(screenNumber);
black         = BlackIndex(screenNumber);
grey          = white / 2;

% Open window
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, grey);
[xCenter, yCenter]    = RectCenter(windowRect);

% Anti‐aliasing
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

% Build the checker‐circle texture (unchanged)
xRadius    = windowRect(3) / 2;
yRadius    = windowRect(4) / 2;
screenYlim = windowRect(4) * .97;
rcycles    = 1;
tcycles    = 2;
xylim      = pi * rcycles;
[x, y]     = meshgrid(-xylim:2*xylim/(screenYlim-1):xylim, ...
                      -xylim:2*xylim/(screenYlim-1):xylim);
at         = atan2(y, x);
checks     = ((1 + sign(sin(at * tcycles) + eps) .* sign(sin(sqrt(x.^2+y.^2))))/2) * (white-black) + black;
circle     = x.^2 + y.^2 <= xylim^2;
checks     = circle .* checks + grey * ~circle;
radialCheckerboardTexture = Screen('MakeTexture', window, checks);

% Pre‐compute destination rectangle
texSize = size(checks,1);
baseRect = [0 0 texSize texSize];
dstRect  = CenterRectOnPointd(baseRect, xCenter, yCenter);

% === New: timing for ping-pong rotation ===
timeTo90 = 2;               % seconds to go from 0° → 90° (tweak this!)
cycleDur = 2 * timeTo90;    % full back & forth
tStart   = GetSecs;

% Initial flip
vbl = Screen('Flip', window);

try
    % Animation loop until keypress
    while ~KbCheck
        % compute elapsed time within cycle
        t = mod(GetSecs - tStart, cycleDur);
        if t <= timeTo90
            angle = (t / timeTo90) * 90;          % ramp up 0→90
        else
            angle = ((cycleDur - t) / timeTo90) * 90;  % ramp down 90→0
        end

        % Draw the rotated texture
        Screen('DrawTexture', window, radialCheckerboardTexture, [], dstRect, angle);

        % Flip
        vbl = Screen('Flip', window, vbl + 0.5 * Screen('GetFlipInterval', window));
    end
catch ME
    sca;
    rethrow(ME);
end

% Clean up
sca;
